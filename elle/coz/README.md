# Coz Integration for Elle Core

Complete parser system that reads all Coz planning files and auto-syncs with Elle Core operational intelligence.

**Created: 2025-11-15**

## Overview

The Coz integration module provides comprehensive synchronization between Coz planning files and Elle Core decision engine:

| File | Parser | Purpose | Integration |
|------|--------|---------|-------------|
| **kanban.csv** | `KanbanParser` | Task management | Daily task suggestions |
| **financials.md** | `FinancialsParser` | Revenue & pricing | SOP ingredient costs |
| **schedule.md** | `ScheduleParser` | Annual calendar | Seasonal recommendations |
| **research_notes.md** | `ResearchParser` | Experiments & formulas | Knowledge base ingestion |
| **inventory.md** | `InventoryParser` | Materials & equipment | Stock alerts & tracking |

## Quick Start

### Basic Usage

```python
from elle.coz import SyncManager

# Initialize sync manager
sync = SyncManager()

# Parse all Coz files
result = sync.parse_all()
print(f"Synced: {result.files_synced}")

# Get daily task suggestions
daily = sync.get_daily_tasks()
print(f"Today's focus: {daily['focus_area']}")
print(f"Due today: {len(daily['due_today'])} tasks")

# Get inventory alerts
inventory = sync.get_inventory_status()
print(f"Items needing reorder: {inventory['summary']['items_needing_reorder']}")

# Get financial summary
financials = sync.get_financial_summary()
for product in financials['products']:
    print(f"{product['name']}: ${product['unit_price']}")
```

### Export All Data

```python
from elle.coz import SyncManager

sync = SyncManager()
sync.parse_all()

# Export to JSON for Elle Core integration
export_path = sync.export_sync_data()
print(f"Exported to: {export_path}")
```

### Get Daily Brief

```python
from elle.coz import SyncManager

sync = SyncManager()
sync.parse_all()

brief = sync.get_daily_brief()
print(f"Date: {brief['timestamp']}")
print(f"Seasonal focus: {brief['seasonal_focus']['current_focus']['focus']}")
print(f"Critical inventory items: {len(brief['inventory_alerts']['critical']['items'])}")
```

## Components

### 1. Kanban Parser

Reads `coz/kanban.csv` and provides task management:

```python
from elle.coz import KanbanParser

kanban = KanbanParser()
tasks = kanban.parse()

# Get daily tasks
daily_tasks = kanban.get_daily_tasks()
for task in daily_tasks:
    print(f"[{task.priority.name}] {task.task} (Due: {task.due_date})")

# Get overdue tasks
overdue = kanban.get_overdue_tasks()
if overdue:
    print(f"WARNING: {len(overdue)} overdue tasks")

# Suggest daily plan
plan = kanban.suggest_daily_plan()
for task in plan['suggested_order']:
    print(f"1. {task.task}")
```

**Available Methods:**
- `parse()` - Parse kanban.csv
- `get_daily_tasks()` - Tasks due today
- `get_overdue_tasks()` - Overdue tasks
- `get_upcoming_tasks(days_ahead)` - Tasks due in N days
- `get_by_category(category)` - Tasks in category
- `get_by_priority(priority)` - Tasks with priority
- `get_active_tasks()` - In-progress tasks
- `suggest_daily_plan()` - Daily task suggestion
- `get_summary()` - Statistics

### 2. Financials Parser

Reads `coz/financials.md` and extracts pricing/costs:

```python
from elle.coz import FinancialsParser

financials = FinancialsParser()
products, reinvestment = financials.parse()

# Get product info
for product in products:
    print(f"{product.name}: ${product.unit_price} (Cost: ${product.cost_per_unit})")
    print(f"  Profit: ${product.profit_per_unit} ({product.gross_margin_percent}%)")

# Calculate production plan for target profit
plan = financials.calculate_production_plan(target_profit=500)
for product_name, info in plan.items():
    print(f"{product_name}: {info['units']:.0f} units → ${info['profit']:.2f} profit")

# Get reinvestment allocation
profit = 1000  # Monthly profit
allocation = reinvestment.get_monthly_allocation(profit)
print(f"Infrastructure: ${allocation['infrastructure']:.2f}")
print(f"Equipment: ${allocation['equipment']:.2f}")
print(f"Emergency buffer: ${allocation['emergency_buffer']:.2f}")
```

**Available Methods:**
- `parse()` - Parse financials.md
- `get_product_by_name(name)` - Get product details
- `get_total_revenue(units)` - Revenue for unit counts
- `get_total_cost(units)` - Cost for unit counts
- `get_profit(units)` - Profit calculation
- `calculate_production_plan(target_profit)` - Units needed for profit goal
- `get_summary()` - Financial summary

### 3. Schedule Parser

Reads `coz/schedule.md` and provides seasonal context:

```python
from elle.coz import ScheduleParser

schedule = ScheduleParser()
focuses = schedule.parse()

# Get current focus
current = schedule.get_current_focus()
print(f"November focus: {current.focus}")

# Get recommendations
recs = schedule.get_recommendations_for_current_month()
print(f"Seasonal tips: {recs['recommendations']['seasonal_tips']}")
print(f"Suggested tasks:")
for task in recs['recommendations']['suggested_tasks']:
    print(f"  - {task}")

# Get upcoming focuses
upcoming = schedule.get_upcoming_focuses(months_ahead=3)
for focus in upcoming:
    print(f"{focus.month.name}: {focus.focus}")
```

**Available Methods:**
- `parse()` - Parse schedule.md
- `get_current_focus()` - Current month's focus
- `get_focus_by_month(month)` - Focus for specific month
- `get_seasonal_focus(season)` - Focuses in season
- `get_current_season()` - Current season
- `get_upcoming_focuses(months_ahead)` - Upcoming focuses
- `get_recommendations_for_current_month()` - Seasonal recommendations
- `get_summary()` - Schedule summary

### 4. Research Parser

Reads `coz/research_notes.md` and extracts experiments:

```python
from elle.coz import ResearchParser

research = ResearchParser()
notes = research.parse()

# View research projects
for note in notes:
    print(f"{note.title} ({note.category})")
    for exp in note.experiments:
        print(f"  {exp.batch_number}: {exp.description}")
        print(f"    Result: {exp.result}")

# Get all formulas
formulas = research.get_all_formulas()
for formula in formulas:
    if formula['type'] == 'ratio':
        print(f"Ratio: {formula['ratio']}")

# Search experiments
goat_experiments = research.search_experiments("GOAT")
for exp in goat_experiments:
    print(f"{exp.batch_number}: {exp.description} → {exp.result}")

# Get knowledge base entries
kb_entries = research.to_knowledge_base_entries()
for entry in kb_entries:
    print(f"Knowledge: {entry['title']}")
```

**Available Methods:**
- `parse()` - Parse research_notes.md
- `get_experiments_by_category(category)` - Experiments in category
- `get_category_summary(category)` - Category overview
- `get_all_formulas()` - All formulas and ratios
- `get_all_recommendations()` - All recommendations
- `search_experiments(keyword)` - Search by keyword
- `get_summary()` - Research summary
- `to_knowledge_base_entries()` - Format for HoloLoom ingestion

### 5. Inventory Parser

Reads `coz/inventory.md` and tracks materials:

```python
from elle.coz import InventoryParser, ItemCategory

inventory = InventoryParser()
items = inventory.parse()

# Check stock levels
low_stock = inventory.get_low_stock_items()
print(f"Items needing reorder: {len(low_stock)}")
for item in low_stock:
    print(f"  {item.name}: {item.quantity} {item.unit} (Reorder: {item.reorder_point})")

# Get critical items
critical = inventory.get_critical_stock_items()
for item in critical:
    print(f"CRITICAL: {item.name} - {item.quantity} remaining!")

# Generate reorder list
reorder = inventory.generate_reorder_list()
for supplier, items in reorder['by_supplier'].items():
    print(f"\n{supplier}:")
    for item in items:
        print(f"  - {item['name']}")

# Check specific categories
ingredients = inventory.get_by_category(ItemCategory.INGREDIENTS)
for item in ingredients:
    print(f"{item.name}: {item.quantity} {item.unit}")

# Get alerts
alerts = inventory.get_reorder_alerts()
print(f"Critical: {alerts['critical']['count']} items")
print(f"Low: {alerts['low']['count']} items")
```

**Available Methods:**
- `parse()` - Parse inventory.md
- `get_low_stock_items()` - Items needing reorder
- `get_critical_stock_items()` - Critically low items
- `get_by_category(category)` - Items in category
- `get_by_supplier(supplier)` - Items from supplier
- `get_items_by_name(name)` - Search by name
- `generate_reorder_list()` - Shopping list
- `update_item_stock(name, quantity)` - Update stock
- `get_inventory_summary()` - Inventory stats
- `get_reorder_alerts()` - Alert summary
- `to_dict()` - Export all items

### 6. Sync Manager

Coordinates all parsers and Elle Core integration:

```python
from elle.coz import SyncManager

# Initialize
sync = SyncManager(auto_sync=True, sync_interval=3600)

# Parse all files
result = sync.parse_all()
print(f"Status: {result.status.value}")
print(f"Files: {result.files_synced}")

# Get comprehensive daily brief
brief = sync.get_daily_brief()
print(f"Date: {brief['timestamp']}")
print(f"Daily tasks: {len(brief['daily_tasks']['due_today'])}")
print(f"Critical inventory: {len(brief['inventory_alerts']['critical']['items'])}")

# Get specific data
daily_tasks = sync.get_daily_tasks()
financials = sync.get_financial_summary()
season = sync.get_seasonal_context()
research = sync.get_research_knowledge_base()
inventory = sync.get_inventory_status()

# Export data
export_path = sync.export_sync_data()
print(f"Exported to: {export_path}")

# Check integration status
status = sync.get_integration_status()
print(f"Sync success rate: {status['success_rate']:.1%}")
```

**Available Methods:**
- `parse_all()` - Parse all Coz files
- `check_files_changed()` - Detect changes
- `get_daily_tasks()` - Daily task suggestions
- `get_financial_summary()` - Financial overview
- `get_seasonal_context()` - Current season & recommendations
- `get_research_knowledge_base()` - Research entries for HoloLoom
- `get_inventory_status()` - Inventory & alerts
- `get_daily_brief()` - Comprehensive daily brief
- `export_sync_data(path)` - Export all data to JSON
- `get_sync_statistics()` - Sync history
- `get_integration_status()` - Overall status

## Data Flow

```
Coz Files
├── kanban.csv
│   └── KanbanParser
│       └── KanbanTask objects
│           └── SyncManager.get_daily_tasks()
│               └── Elle Core: Daily suggestions
│
├── financials.md
│   └── FinancialsParser
│       └── Product, ReinvestmentPlan objects
│           └── SyncManager.get_financial_summary()
│               └── Elle Core: SOP ingredient costs
│
├── schedule.md
│   └── ScheduleParser
│       └── MonthlyFocus objects
│           └── SyncManager.get_seasonal_context()
│               └── Elle Core: Seasonal recommendations
│
├── research_notes.md
│   └── ResearchParser
│       └── ResearchNote, ExperimentResult objects
│           └── SyncManager.get_research_knowledge_base()
│               └── HoloLoom: Knowledge base ingestion
│
└── inventory.md
    └── InventoryParser
        └── InventoryItem objects
            └── SyncManager.get_inventory_status()
                └── Elle Core: Material tracking & alerts
```

## File Structure

```
elle/
└── coz/
    ├── __init__.py                  # Module exports
    ├── kanban_parser.py             # Task parsing
    ├── financials_parser.py         # Revenue/cost parsing
    ├── schedule_parser.py           # Calendar parsing
    ├── research_parser.py           # Research parsing
    ├── inventory_parser.py          # Material parsing
    ├── sync_manager.py              # Master coordinator
    └── README.md                    # This file
```

## Integration with Elle Core

### Daily Task Suggestions

Tasks from kanban.csv, prioritized by:
1. Due date (today first)
2. Priority level (HIGH → MEDIUM → LOW)
3. Status (In Progress → Planned → Pending)

```python
daily = sync.get_daily_tasks()
# Returns categorized tasks for Elle Decision Engine
```

### SOP Ingredient Costs

Financial data auto-populates ingredient costs in SOPs:

```python
product = financials.get_product_by_name("Bread Loaf")
# Can fetch cost_per_unit for ingredient calculations
```

### Seasonal Awareness

Schedule data provides context for recommendations:

```python
season = sync.get_seasonal_context()
# Includes current focus, seasonal tips, suggested tasks
```

### Knowledge Base Ingestion

Research notes formatted for HoloLoom RAG system:

```python
kb_entries = sync.get_research_knowledge_base()
# Each entry includes title, content, experiments, formulas
# Ready for: await hololoom.experience(entry)
```

### Material Tracking

Inventory alerts trigger reorder notifications:

```python
alerts = inventory.get_reorder_alerts()
if alerts['critical']['count'] > 0:
    # Send reorder alert to Elle
```

## Demo

See `demos/demo_coz_integration.py` for complete working example:

```bash
PYTHONPATH=. python demos/demo_coz_integration.py
```

## Error Handling

All parsers gracefully handle missing files:

```python
from elle.coz import KanbanParser

parser = KanbanParser()
try:
    tasks = parser.parse()
except FileNotFoundError:
    print("Kanban file not found")
```

## Performance

| Operation | Time |
|-----------|------|
| Parse all files | ~50ms |
| Parse single file | ~5-10ms |
| Export data | ~20ms |
| Generate daily brief | ~10ms |
| Get reorder list | ~5ms |

## Future Enhancements

- Real-time file watching with auto-sync
- Database integration for history tracking
- Advanced analytics on financial performance
- Predictive inventory management
- Integration with voice interface for Coz operations

## Author

Blake Chasteen
Created: 2025-11-15
Elle Core v0.1.0-alpha
