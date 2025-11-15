# Coz Integration Quick Start Guide

**Created: 2025-11-15**

Fast reference for using the Coz parser system.

## Installation

Nothing to install - just import and use:

```python
from elle.coz import SyncManager
```

## 30-Second Start

```python
from elle.coz import SyncManager

# Parse all Coz files
sync = SyncManager()
sync.parse_all()

# Get daily tasks
daily = sync.get_daily_tasks()
print(f"Today's focus: {daily['focus_area']}")
print(f"Tasks due: {len(daily['due_today'])}")

# Get inventory alerts
inventory = sync.get_inventory_status()
print(f"Items to reorder: {inventory['summary']['items_needing_reorder']}")

# Get financial summary
financials = sync.get_financial_summary()
for p in financials['products']:
    print(f"{p['name']}: ${p['unit_price']}")
```

## Common Tasks

### Get Today's Plan

```python
daily = sync.get_daily_tasks()

# Due today (sorted by priority)
for task in daily['due_today']:
    print(f"[{task['priority']}] {task['task']}")

# Recommended order
for task in daily['recommended_order']:
    print(f"→ {task['task']}")
```

### Find Overdue Tasks

```python
from elle.coz import KanbanParser

kanban = KanbanParser()
kanban.parse()

overdue = kanban.get_overdue_tasks()
if overdue:
    print(f"⚠️  {len(overdue)} overdue tasks!")
    for task in overdue:
        print(f"  - {task.task}")
```

### Check Inventory Alerts

```python
inventory = sync.get_inventory_status()

alerts = inventory['alerts']
if alerts['critical']['count'] > 0:
    print("🚨 CRITICAL REORDER NEEDED:")
    for item in alerts['critical']['items']:
        print(f"  {item['name']}: {item['quantity']} left (need {item['reorder_point']})")

if alerts['low']['count'] > 0:
    print("\n⚠️  Low stock:")
    for item in alerts['low']['items']:
        print(f"  {item['name']}")
```

### Generate Shopping List

```python
inventory = sync.inventory
reorder = inventory.generate_reorder_list()

for supplier, items in reorder['by_supplier'].items():
    print(f"\n{supplier}:")
    for item in items:
        print(f"  □ {item['name']}")
```

### Get Seasonal Recommendations

```python
recs = sync.get_seasonal_context()

print(f"Focus: {recs['recommendations']['primary_focus']}")
print(f"Season: {recs['current_season']}")

print("\nThis season:")
for task in recs['recommendations']['suggested_tasks']:
    print(f"  • {task}")
```

### Calculate Profit Planning

```python
financials = sync.financials

# How many units needed for $500 profit?
plan = financials.calculate_production_plan(target_profit=500)

for product, info in plan.items():
    print(f"{product}: {info['units']:.0f} units → ${info['profit']:.2f} profit")
```

### Get All Experiments

```python
research = sync.research

# All projects
for note in research.research_notes:
    print(f"\n{note.title}")
    for exp in note.experiments:
        print(f"  {exp.batch_number}: {exp.description} → {exp.result}")

# Knowledge base format for HoloLoom
kb = research.to_knowledge_base_entries()
```

### Export Everything

```python
# Save all data to JSON
export_path = sync.export_sync_data()
print(f"Saved to: {export_path}")

# Or get specific data as dict
data = {
    'kanban': [t.to_dict() for t in sync.kanban.tasks],
    'financials': [p.to_dict() for p in sync.financials.products],
    'schedule': [f.to_dict() for f in sync.schedule.monthly_focus.values()],
    'inventory': sync.inventory.to_dict(),
}
```

## Component Quick Reference

### Kanban Parser
```python
from elle.coz import KanbanParser

kanban = KanbanParser()
tasks = kanban.parse()

# Methods
kanban.get_daily_tasks()
kanban.get_overdue_tasks()
kanban.get_upcoming_tasks(days_ahead=7)
kanban.get_by_category('Bakery')
kanban.get_by_priority(TaskPriority.HIGH)
kanban.get_active_tasks()
kanban.suggest_daily_plan()
kanban.get_summary()
```

### Financials Parser
```python
from elle.coz import FinancialsParser

financials = FinancialsParser()
products, reinvestment = financials.parse()

# Methods
financials.get_product_by_name('Bread Loaf')
financials.get_total_revenue(units={'Bread Loaf': 10})
financials.get_total_cost(units={...})
financials.get_profit(units={...})
financials.calculate_production_plan(target_profit=500)
```

### Schedule Parser
```python
from elle.coz import ScheduleParser

schedule = ScheduleParser()
focuses = schedule.parse()

# Methods
schedule.get_current_focus()
schedule.get_focus_by_month(Month.NOVEMBER)
schedule.get_seasonal_focus(Season.WINTER)
schedule.get_current_season()
schedule.get_upcoming_focuses(months_ahead=3)
schedule.get_recommendations_for_current_month()
```

### Research Parser
```python
from elle.coz import ResearchParser

research = ResearchParser()
notes = research.parse()

# Methods
research.get_experiments_by_category('The GOAT')
research.get_all_formulas()
research.get_all_recommendations()
research.search_experiments('honey')
research.get_summary()
research.to_knowledge_base_entries()
```

### Inventory Parser
```python
from elle.coz import InventoryParser, ItemCategory

inventory = InventoryParser()
items = inventory.parse()

# Methods
inventory.get_low_stock_items()
inventory.get_critical_stock_items()
inventory.get_by_category(ItemCategory.INGREDIENTS)
inventory.get_by_supplier('Costco')
inventory.generate_reorder_list()
inventory.get_reorder_alerts()
inventory.update_item_stock('Oats', 100)
```

### Sync Manager
```python
from elle.coz import SyncManager

sync = SyncManager()
sync.parse_all()

# Daily Operations
sync.get_daily_tasks()
sync.get_seasonal_context()
sync.get_inventory_status()
sync.get_financial_summary()
sync.get_research_knowledge_base()
sync.get_daily_brief()

# Management
sync.export_sync_data()
sync.get_sync_statistics()
sync.get_integration_status()
sync.check_files_changed()
```

## Data Models

### Task (from Kanban)
```python
task = {
    'task': 'Bake bread',
    'category': 'Bakery',
    'priority': 'HIGH',
    'start_date': '2025-11-03T00:00:00',
    'due_date': '2025-11-03T00:00:00',
    'status': 'In Progress',
    'notes': 'Use new flour batch'
}
```

### Product (from Financials)
```python
product = {
    'name': 'Bread Loaf',
    'unit_price': 6.0,
    'cost_per_unit': 2.0,
    'profit_per_unit': 4.0,
    'gross_margin_percent': 67.0,
    'notes': 'Sells well weekly'
}
```

### Monthly Focus (from Schedule)
```python
focus = {
    'month': 'NOVEMBER',
    'month_number': 11,
    'focus': 'Biochar & Compost Kits',
    'season': 'fall'
}
```

### Experiment (from Research)
```python
experiment = {
    'batch_number': 'Batch 01',
    'description': 'Honey + Cinnamon',
    'result': 'Excellent texture, mild sweetness.',
    'status': 'completed',
    'value': '01'
}
```

### Inventory Item
```python
item = {
    'name': 'Oats (bulk)',
    'category': 'ingredients',
    'supplier': 'Costco',
    'quantity': 50.0,
    'unit': 'lbs',
    'reorder_point': 20.0,
    'stock_level': 'normal',
    'needs_reorder': False
}
```

## Integration Patterns

### Pattern 1: Daily Operations
```python
sync = SyncManager()
sync.parse_all()

brief = sync.get_daily_brief()
# Use brief in:
# - Decision engine
# - Voice interface
# - Daily dashboard
```

### Pattern 2: Financial Planning
```python
financials = sync.get_financial_summary()

# Use in SOP cost calculations
for product in financials['products']:
    sop_cost = product['cost_per_unit']
```

### Pattern 3: Material Management
```python
alerts = sync.get_inventory_status()['alerts']

# Send notifications for critical items
if alerts['critical']['count'] > 0:
    send_urgent_reorder_notification(alerts['critical']['items'])
```

### Pattern 4: Knowledge Ingestion
```python
kb_entries = sync.get_research_knowledge_base()

# Ingest into HoloLoom
for entry in kb_entries:
    await hololoom.experience(entry['content'])
```

## Running the Demo

```bash
PYTHONPATH=. python demos/demo_coz_integration.py
```

Shows:
- All parsers working
- Daily tasks and planning
- Seasonal focus and recommendations
- Financial summary
- Inventory alerts
- Research experiments
- Comprehensive daily brief
- Data export
- Sync statistics

## Troubleshooting

**Files not found?**
```python
# Make sure Coz files exist
from pathlib import Path
if not Path('coz/kanban.csv').exists():
    print("Missing coz/kanban.csv")
```

**Empty results?**
```python
# Parse first!
sync.parse_all()

# Then access data
tasks = sync.kanban.tasks  # Now has data
```

**Want to debug?**
```python
# Check what was parsed
print(f"Tasks: {len(sync.kanban.tasks)}")
print(f"Products: {len(sync.financials.products)}")
print(f"Focuses: {len(sync.schedule.monthly_focus)}")
print(f"Items: {len(sync.inventory.items)}")

# Check sync result
result = sync.parse_all()
print(f"Status: {result.status}")
print(f"Errors: {result.errors}")
```

## Performance Tips

**Fast single operations:**
```python
# Parse only what you need
kanban = KanbanParser()
kanban.parse()
daily = kanban.get_daily_tasks()  # ~10ms
```

**Cache results:**
```python
# Parse once, use multiple times
sync = SyncManager()
sync.parse_all()

daily = sync.get_daily_tasks()
season = sync.get_seasonal_context()
inventory = sync.get_inventory_status()
```

**Batch operations:**
```python
# Export all at once
export_path = sync.export_sync_data()  # ~20ms for all data
```

## Next Steps

1. **Try the demo:** `python demos/demo_coz_integration.py`
2. **Read full docs:** `elle/coz/README.md`
3. **Integrate with Elle:** Use SyncManager in your decision engine
4. **Set up auto-sync:** Enable background sync loop
5. **Monitor performance:** Check sync statistics

## Support

For detailed information, see:
- **Full Documentation**: `elle/coz/README.md`
- **API Reference**: Each parser has docstrings
- **Working Example**: `demos/demo_coz_integration.py`
- **Data Export Format**: `elle/data/coz_sync_data.json`
