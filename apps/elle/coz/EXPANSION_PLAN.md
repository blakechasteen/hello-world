# COZ Expansion Plan: Time, Cost & Intelligence Layer

**Created:** 2025-11-21
**Status:** Design Complete → Ready for Implementation
**Scope:** Aggressive expansion (7 new parsers + intelligence layer)

---

## 🎯 Executive Summary

Transform COZ from a planning file parser into a **complete operational intelligence system** with:
- ✅ **Time & Cost Tracking** (CORE: track actual vs. planned, profit analysis)
- ✅ **Customer Orders** (revenue pipeline, fulfillment tracking)
- ✅ **SOPs** (standard operating procedures, process templates)
- ✅ **Performance Layer** (scale to 100K+ records, <100ms queries)
- ✅ **Intelligence Layer** (cross-parser insights, actionable recommendations)
- ✅ **Impressive Daily Brief** (executive dashboard with insights)

**Timeline:** 3 phases over 6-8 weeks
**Backward Compatibility:** 100% maintained
**Dependencies:** Zero new external libraries

---

## 1. CURRENT STATE ANALYSIS

### Architecture Overview

**Strengths to Preserve:**
- ✅ Clean parser protocol (consistent `parse()` → data model pattern)
- ✅ SyncManager orchestration (single entry point)
- ✅ Zero external dependencies (pure Python + stdlib)
- ✅ Fast performance (~50ms for all 5 parsers)
- ✅ Complete data models (dataclasses with `to_dict()`)

**Current Components:**
```
elle/coz/
├── kanban_parser.py          (8.7 KB, 260 lines)
├── financials_parser.py      (9.0 KB, 270 lines)
├── schedule_parser.py        (11.5 KB, 340 lines)
├── research_parser.py        (11.4 KB, 340 lines)
├── inventory_parser.py       (11.6 KB, 350 lines)
├── sync_manager.py           (11.4 KB, 298 lines)
└── Total: 63.6 KB, ~1,858 lines
```

**Pain Points Identified:**

1. **Missing Time Tracking**
   - No way to track actual hours worked on tasks
   - Can't compare estimated vs. actual time
   - Can't calculate true hourly profit

2. **Missing Cost Tracking**
   - Product costs exist, but not task/project costs
   - Can't track material waste vs. planned
   - No cost rollup from tasks → projects

3. **No Customer Orders**
   - Revenue is theoretical (unit prices), not actual
   - Can't track pending orders or fulfillment
   - No customer pipeline visibility

4. **No SOPs**
   - Process knowledge is scattered
   - Can't estimate task time based on SOP
   - No standard cost templates

5. **Limited Intelligence**
   - Parsers work in isolation
   - No cross-file insights (e.g., "High inventory of X, but no tasks scheduled")
   - Daily brief is basic aggregation, not insights

6. **Scale Concerns**
   - All data loaded in memory (not sustainable for 100K+ records)
   - No indexing or caching
   - Export to JSON is full dump (no incremental)

**Performance Baseline:**
```
Current (5 parsers, ~500 total records):
- parse_all(): 50ms
- get_daily_brief(): 10ms
- export_sync_data(): 20ms
- Memory: ~5MB

Target (12 parsers, 10K+ records):
- parse_all(): <100ms
- get_daily_brief(): <50ms
- export_sync_data(): <100ms
- Memory: <50MB
```

---

## 2. PROPOSED EXPANSIONS

### 2.1 NEW PARSERS (7 Total)

#### Parser #6: Time Tracking 🕐 (CORE)

**File:** `coz/time_tracking.csv`
**Purpose:** Track actual hours worked on tasks vs. estimates

**Schema:**
```csv
Date,Task ID,Task Name,Category,Estimated Hours,Actual Hours,Notes
2025-11-20,KB-001,Bake bread,Bakery,2.0,2.5,Mixing took longer
2025-11-20,KB-002,Package orders,Fulfillment,1.0,0.75,Efficient packing
2025-11-21,KB-003,Research new recipe,R&D,3.0,4.5,Multiple iterations
```

**Data Model:**
```python
@dataclass
class TimeEntry:
    date: datetime
    task_id: str
    task_name: str
    category: str
    estimated_hours: float
    actual_hours: float
    notes: str
    variance_hours: float  # actual - estimated
    variance_percent: float  # (actual - estimated) / estimated
    efficiency_score: float  # estimated / actual (1.0 = on time)
```

**Key Methods:**
```python
parser = TimeTrackingParser()
entries = parser.parse()

# Analysis
parser.get_time_summary()  # Total estimated vs. actual
parser.get_by_category('Bakery')  # Category breakdown
parser.get_efficiency_by_task()  # Which tasks consistently overrun?
parser.get_weekly_summary()  # Hours worked this week
parser.get_hourly_cost(hourly_rate=25)  # Labor cost calculation
parser.predict_task_time(task_name='Bake bread')  # Historical average
```

**Integration Points:**
- **Kanban:** Link time entries to tasks (task_id → KanbanTask)
- **Financials:** Calculate true hourly profit (revenue - costs - labor)
- **Intelligence Layer:** Detect chronic time overruns, recommend re-estimation

**Implementation Effort:** 2-3 days

---

#### Parser #7: Cost Tracking 💰 (CORE)

**File:** `coz/cost_tracking.csv`
**Purpose:** Track actual costs per task/project (materials, labor, overhead)

**Schema:**
```csv
Date,Task ID,Task Name,Material Cost,Labor Cost,Overhead Cost,Total Cost,Notes
2025-11-20,KB-001,Bake bread,2.50,62.50,5.00,70.00,Used premium flour
2025-11-20,KB-002,Package orders,0.50,18.75,2.00,21.25,
2025-11-21,KB-003,Research new recipe,8.00,112.50,10.00,130.50,Multiple test batches
```

**Data Model:**
```python
@dataclass
class CostEntry:
    date: datetime
    task_id: str
    task_name: str
    material_cost: float
    labor_cost: float
    overhead_cost: float
    total_cost: float
    notes: str

    # Calculated fields
    cost_per_hour: float  # total_cost / hours_worked (from time tracking)
    profit_margin: float  # (revenue - total_cost) / revenue (from financials)
```

**Key Methods:**
```python
parser = CostTrackingParser()
entries = parser.parse()

# Analysis
parser.get_cost_summary()  # Total costs by category
parser.get_by_task('KB-001')  # All costs for a task
parser.get_monthly_costs()  # Spending trends
parser.get_cost_variance(task_id, estimated_cost)  # Budget vs. actual
parser.get_profit_by_task()  # Revenue - cost for each task
parser.get_overhead_rate()  # Overhead as % of direct costs
```

**Integration Points:**
- **Time Tracking:** Calculate labor costs (hours × rate)
- **Financials:** Compare actual costs vs. product cost_per_unit
- **Kanban:** Budget tracking per task
- **Intelligence Layer:** Detect cost overruns, optimize material usage

**Implementation Effort:** 2-3 days

---

#### Parser #8: Customer Orders 📦

**File:** `coz/customer_orders.csv`
**Purpose:** Track revenue pipeline, order fulfillment, customer data

**Schema:**
```csv
Order ID,Customer Name,Order Date,Due Date,Status,Products,Quantities,Total Price,Notes
ORD-001,Alice Smith,2025-11-18,2025-11-20,Fulfilled,"Bread Loaf,Cookies","2,12",26.00,Weekly regular
ORD-002,Bob Jones,2025-11-19,2025-11-22,In Progress,Bread Loaf,5,30.00,First-time customer
ORD-003,Carol White,2025-11-20,2025-11-25,Pending,Custom Cake,1,45.00,Birthday cake
```

**Data Model:**
```python
@dataclass
class CustomerOrder:
    order_id: str
    customer_name: str
    order_date: datetime
    due_date: datetime
    status: OrderStatus  # PENDING, IN_PROGRESS, FULFILLED, CANCELLED
    products: List[str]
    quantities: List[int]
    total_price: float
    notes: str

    # Calculated
    days_until_due: int
    is_overdue: bool
    fulfillment_priority: int  # 1-5 based on due date + customer importance
```

**Key Methods:**
```python
parser = CustomerOrdersParser()
orders = parser.parse()

# Analysis
parser.get_pending_orders()  # Orders needing fulfillment
parser.get_revenue_pipeline()  # Total pending revenue
parser.get_due_this_week()  # Orders due in 7 days
parser.get_overdue_orders()  # Late orders
parser.get_customer_summary('Alice Smith')  # Customer history
parser.get_fulfillment_schedule()  # Recommended production order
parser.get_revenue_forecast(days_ahead=30)  # Projected revenue
```

**Integration Points:**
- **Financials:** Actual revenue vs. theoretical unit prices
- **Kanban:** Auto-create tasks from orders
- **Inventory:** Check stock levels against orders
- **Intelligence Layer:** Predict stockouts, recommend production

**Implementation Effort:** 3-4 days

---

#### Parser #9: SOPs (Standard Operating Procedures) 📋

**File:** `coz/sops.md`
**Purpose:** Store process templates with steps, time estimates, cost estimates

**Schema (Markdown):**
```markdown
## SOP: Bake Bread Loaf

**Category:** Bakery
**Estimated Time:** 2.0 hours
**Estimated Cost:** $2.50 (materials) + $50 (labor @ $25/hr) = $52.50
**Difficulty:** Medium
**Output:** 1 bread loaf

### Steps:
1. Mix dry ingredients (15 min)
2. Add wet ingredients, knead (20 min)
3. First rise (60 min)
4. Shape loaf (10 min)
5. Second rise (30 min)
6. Bake at 350°F (25 min)

### Materials:
- Flour: 500g
- Water: 350ml
- Yeast: 10g
- Salt: 10g

### Notes:
- Use room temperature water
- Cover during rises
```

**Data Model:**
```python
@dataclass
class SOP:
    title: str
    category: str
    estimated_time_hours: float
    estimated_cost: float
    difficulty: str
    output_description: str
    steps: List[str]
    materials: Dict[str, str]
    notes: str
```

**Key Methods:**
```python
parser = SOPParser()
sops = parser.parse()

# Retrieval
parser.get_by_name('Bake Bread Loaf')
parser.get_by_category('Bakery')
parser.search_steps('knead')  # Find SOPs with specific steps

# Analysis
parser.estimate_task_time(task_name='Bake bread')  # Lookup SOP time
parser.estimate_task_cost(task_name='Bake bread')  # Lookup SOP cost
parser.get_materials_needed(sop_name='Bake Bread Loaf')  # Materials list
```

**Integration Points:**
- **Kanban:** Auto-populate time estimates from SOPs
- **Time Tracking:** Compare actual vs. SOP estimates
- **Cost Tracking:** Compare actual vs. SOP cost estimates
- **Inventory:** Check stock for SOP materials
- **Intelligence Layer:** Recommend SOP creation for frequent tasks

**Implementation Effort:** 3-4 days

---

#### Parser #10: Production Log 🏭

**File:** `coz/production_log.csv`
**Purpose:** Track what was produced, when, quantities, waste

**Schema:**
```csv
Date,Product,Quantity Produced,Quantity Sold,Quantity Wasted,Waste Reason,Notes
2025-11-20,Bread Loaf,10,8,2,Overproduction,Made too many
2025-11-20,Cookies,24,24,0,,Sold out
2025-11-21,Custom Cake,1,1,0,,Birthday order
```

**Data Model:**
```python
@dataclass
class ProductionEntry:
    date: datetime
    product: str
    quantity_produced: int
    quantity_sold: int
    quantity_wasted: int
    waste_reason: str
    notes: str

    # Calculated
    waste_percentage: float
    revenue_actual: float  # quantity_sold × unit_price
    revenue_lost: float  # quantity_wasted × unit_price
```

**Key Methods:**
```python
parser = ProductionLogParser()
entries = parser.parse()

# Analysis
parser.get_production_summary()  # Total produced/sold/wasted
parser.get_waste_analysis()  # Waste by reason
parser.get_revenue_actual()  # Actual revenue from sales
parser.get_revenue_lost_to_waste()  # Money lost to waste
parser.predict_production(product='Bread Loaf', days_ahead=7)  # Historical avg
```

**Integration Points:**
- **Financials:** Actual revenue vs. theoretical
- **Customer Orders:** Fulfill orders from production
- **Inventory:** Update stock based on production
- **Intelligence Layer:** Optimize production quantities to minimize waste

**Implementation Effort:** 2-3 days

---

#### Parser #11: Expenses 💸

**File:** `coz/expenses.csv`
**Purpose:** Track one-time expenses, subscriptions, overhead

**Schema:**
```csv
Date,Category,Description,Amount,Frequency,Vendor,Notes
2025-11-15,Equipment,Stand mixer,$250,One-time,KitchenAid,
2025-11-01,Subscriptions,Cloud storage,$10,Monthly,Dropbox,
2025-11-10,Utilities,Electricity,$75,Monthly,PG&E,
2025-11-12,Marketing,Instagram ads,$50,One-time,Meta,
```

**Data Model:**
```python
@dataclass
class Expense:
    date: datetime
    category: ExpenseCategory  # EQUIPMENT, SUBSCRIPTIONS, UTILITIES, MARKETING, etc.
    description: str
    amount: float
    frequency: str  # ONE_TIME, MONTHLY, ANNUAL
    vendor: str
    notes: str
```

**Key Methods:**
```python
parser = ExpensesParser()
expenses = parser.parse()

# Analysis
parser.get_monthly_expenses()  # Total expenses this month
parser.get_by_category('Equipment')  # Category breakdown
parser.get_recurring_expenses()  # Monthly/annual subscriptions
parser.get_expense_forecast(months_ahead=6)  # Projected expenses
parser.calculate_overhead_rate()  # Overhead as % of revenue
```

**Integration Points:**
- **Cost Tracking:** Include overhead in task costs
- **Financials:** Calculate true profit (revenue - costs - expenses)
- **Intelligence Layer:** Detect unnecessary subscriptions, optimize expenses

**Implementation Effort:** 2 days

---

#### Parser #12: Suppliers 🚚

**File:** `coz/suppliers.md`
**Purpose:** Track supplier contacts, pricing, lead times, order history

**Schema (Markdown):**
```markdown
## Costco

**Category:** Bulk Ingredients
**Contact:** 1-800-COSTCO
**Lead Time:** 1 day (in-store pickup)
**Min Order:** $100
**Payment Terms:** Cash/Credit

### Products:
- Flour (25 lb): $15
- Sugar (10 lb): $8
- Butter (4 lb): $12

### Order History:
- 2025-11-01: Flour × 2, Sugar × 1 = $38
- 2025-10-15: Butter × 3, Flour × 1 = $51
```

**Data Model:**
```python
@dataclass
class Supplier:
    name: str
    category: str
    contact: str
    lead_time_days: int
    min_order: float
    payment_terms: str
    products: Dict[str, float]  # product → price
    order_history: List[Dict]
```

**Key Methods:**
```python
parser = SuppliersParser()
suppliers = parser.parse()

# Retrieval
parser.get_by_name('Costco')
parser.get_by_category('Bulk Ingredients')
parser.find_product('Flour')  # Which suppliers sell this?

# Analysis
parser.get_cheapest_supplier(product='Flour')
parser.get_fastest_supplier(product='Flour')  # By lead time
parser.calculate_order_total(supplier='Costco', products={'Flour': 2, 'Sugar': 1})
parser.get_order_recommendations()  # Based on inventory reorder list
```

**Integration Points:**
- **Inventory:** Link items to suppliers, pricing, lead times
- **Expenses:** Track supplier orders as expenses
- **Intelligence Layer:** Recommend cheapest/fastest supplier for reorder

**Implementation Effort:** 2-3 days

---

### 2.2 PERFORMANCE OPTIMIZATIONS

#### Optimization #1: Lazy Loading

**Problem:** All parsers load full data in memory on `parse()`
**Solution:** Lazy load only when accessed

**Implementation:**
```python
class LazyParser:
    def __init__(self):
        self._data = None
        self._parsed = False

    @property
    def data(self):
        if not self._parsed:
            self.parse()
        return self._data
```

**Expected Speedup:** 3x for `SyncManager.__init__()` (no upfront parsing)

---

#### Optimization #2: Incremental Sync

**Problem:** `export_sync_data()` dumps all data every time
**Solution:** Track changes, export only deltas

**Implementation:**
```python
class IncrementalSyncManager:
    def export_incremental(self, since: datetime) -> Dict:
        """Export only records changed since timestamp"""
        return {
            'kanban': self.kanban.get_changed_since(since),
            'financials': self.financials.get_changed_since(since),
            # ... etc
        }
```

**Expected Speedup:** 10x for export with <10% changed data

---

#### Optimization #3: Indexing

**Problem:** Lookups are O(n) linear scans
**Solution:** Build indexes on common lookup keys

**Implementation:**
```python
class IndexedParser:
    def __init__(self):
        self._data = []
        self._index_by_id = {}  # task_id → task
        self._index_by_date = {}  # date → List[tasks]

    def parse(self):
        self._data = self._load_file()
        self._build_indexes()

    def get_by_id(self, task_id):
        return self._index_by_id.get(task_id)  # O(1)
```

**Expected Speedup:** 100x for ID lookups (O(n) → O(1))

---

#### Optimization #4: Batch Operations

**Problem:** Many small file reads are slow
**Solution:** Read all files in one pass

**Implementation:**
```python
class BatchSyncManager:
    def parse_all_batch(self):
        """Read all files in parallel"""
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.kanban.parse),
                executor.submit(self.financials.parse),
                # ... etc
            }
            wait(futures)
```

**Expected Speedup:** 2x for `parse_all()` via parallelism

---

#### Optimization #5: Caching

**Problem:** Repeated calculations (e.g., `get_daily_tasks()`)
**Solution:** Cache computed results with TTL

**Implementation:**
```python
from functools import lru_cache
from datetime import timedelta

class CachedSyncManager:
    @lru_cache(maxsize=128)
    def get_daily_tasks(self, date: datetime):
        """Cache daily tasks for 5 minutes"""
        # ... computation
```

**Expected Speedup:** 100x for cached queries

---

### 2.3 INTELLIGENCE LAYER 🧠

#### Intelligence Module Architecture

**File:** `elle/coz/intelligence.py`
**Purpose:** Cross-parser insights, recommendations, predictions

**Components:**

1. **Insight Engine** - Detect patterns across parsers
2. **Recommendation Engine** - Suggest actions
3. **Prediction Engine** - Forecast trends
4. **Alert Engine** - Proactive warnings

---

#### Feature #1: Profit Analysis

**Cross-parser:** Time + Cost + Financials + Orders

**Insights:**
```python
intelligence = IntelligenceEngine(sync_manager)

profit = intelligence.analyze_profit()
# {
#   'total_revenue': 1500,  # From orders
#   'total_costs': 800,  # From cost tracking
#   'total_labor': 500,  # From time tracking
#   'total_expenses': 100,  # From expenses
#   'net_profit': 100,
#   'profit_margin': 6.7%,
#   'hourly_profit': 5.00,  # profit / hours_worked
#   'breakeven_hours': 20,  # hours needed to cover costs
# }
```

**Recommendations:**
- "Reduce labor on 'Packaging' (2.5 hrs actual vs. 1.0 est)"
- "Increase price on 'Cookies' (hourly profit: $8 vs. target $15)"

---

#### Feature #2: Production Optimization

**Cross-parser:** Orders + Production + Inventory + Financials

**Insights:**
```python
optimization = intelligence.optimize_production()
# {
#   'recommendations': [
#     {
#       'product': 'Bread Loaf',
#       'recommended_quantity': 12,
#       'reason': '10 pending orders + 20% buffer',
#       'expected_waste': 0,
#       'expected_revenue': 72.00,
#     },
#     {
#       'product': 'Cookies',
#       'recommended_quantity': 0,
#       'reason': 'No pending orders, high waste rate (30%)',
#     }
#   ],
#   'total_expected_revenue': 72.00,
#   'total_expected_waste': 0,
# }
```

---

#### Feature #3: Task Prioritization

**Cross-parser:** Kanban + Orders + Time + Cost

**Insights:**
```python
priorities = intelligence.prioritize_tasks()
# [
#   {
#     'task': 'Fulfill ORD-002',
#     'priority_score': 95,
#     'reasons': [
#       'Due tomorrow (high urgency)',
#       '$30 revenue (high value)',
#       '2 hrs estimated (quick win)',
#     ],
#   },
#   {
#     'task': 'Reorder flour',
#     'priority_score': 85,
#     'reasons': [
#       'Critical stock (0 lbs remaining)',
#       'Blocking 3 pending orders',
#       'Supplier lead time: 1 day',
#     ],
#   },
# ]
```

---

#### Feature #4: Waste Detection

**Cross-parser:** Production + Inventory + Financials

**Insights:**
```python
waste = intelligence.detect_waste()
# {
#   'production_waste': {
#     'Bread Loaf': {
#       'waste_rate': 20%,
#       'revenue_lost': 12.00,
#       'recommendation': 'Reduce production by 2 units',
#     },
#   },
#   'inventory_waste': {
#     'Expired flour': {
#       'quantity': 5 lbs,
#       'cost': 3.00,
#       'recommendation': 'Use FIFO, reduce order quantity',
#     },
#   },
#   'time_waste': {
#     'Packaging': {
#       'time_overrun': 1.5 hrs/week,
#       'cost': 37.50,
#       'recommendation': 'Create SOP, use packing template',
#     },
#   },
# }
```

---

#### Feature #5: Predictive Analytics

**Cross-parser:** All parsers + historical trends

**Insights:**
```python
predictions = intelligence.predict_next_30_days()
# {
#   'revenue_forecast': 2500,  # Based on order trends
#   'cost_forecast': 1200,  # Based on historical costs
#   'profit_forecast': 1300,
#   'tasks_predicted': 45,  # Based on historical workload
#   'hours_predicted': 120,  # Based on time tracking
#   'inventory_reorders': [
#     {'item': 'Flour', 'date': '2025-12-05', 'reason': 'Stock depletion'},
#   ],
#   'risks': [
#     'Production capacity: 80% utilized (bottleneck risk)',
#     'Customer order rate increasing 15% (may need to scale)',
#   ],
# }
```

---

### 2.4 IMPRESSIVE DAILY BRIEF 📊

**File:** `elle/coz/daily_brief.py`
**Purpose:** Executive dashboard with insights, not just data

**Structure:**

```python
def generate_daily_brief(sync: SyncManager, intelligence: IntelligenceEngine) -> Dict:
    """Generate comprehensive daily brief"""

    return {
        # 1. EXECUTIVE SUMMARY
        'executive_summary': {
            'date': '2025-11-21',
            'key_metrics': {
                'revenue_today': 150,
                'profit_today': 45,
                'hours_worked_today': 6.5,
                'tasks_completed': 5,
                'orders_fulfilled': 3,
            },
            'highlights': [
                '✅ Exceeded daily revenue target by 15%',
                '⚠️ Labor cost higher than planned (30% vs. 25% target)',
                '✅ Zero production waste today',
            ],
            'concerns': [
                '🚨 Critical stock: Flour (0 lbs remaining)',
                '⚠️ 2 orders overdue',
            ],
        },

        # 2. TODAY'S PRIORITIES
        'priorities': [
            {
                'task': 'Reorder flour (CRITICAL)',
                'urgency': 'IMMEDIATE',
                'reason': 'Blocking 3 pending orders worth $75',
                'action': 'Call Costco (1-800-COSTCO), pickup today',
            },
            {
                'task': 'Fulfill ORD-002 (Due tomorrow)',
                'urgency': 'HIGH',
                'reason': '$30 revenue, first-time customer',
                'action': 'Bake 5 bread loaves (SOP: 2 hrs)',
            },
            {
                'task': 'Package ORD-001 (Overdue by 1 day)',
                'urgency': 'HIGH',
                'reason': 'Regular customer, maintain relationship',
                'action': 'Package & notify customer',
            },
        ],

        # 3. REVENUE PIPELINE
        'revenue_pipeline': {
            'pending_orders': 5,
            'total_value': 185,
            'due_this_week': 3,
            'overdue': 2,
            'forecast_this_month': 2500,
        },

        # 4. FINANCIAL SNAPSHOT
        'financial_snapshot': {
            'revenue_mtd': 1200,
            'costs_mtd': 650,
            'profit_mtd': 550,
            'profit_margin': 45.8,
            'hourly_profit': 12.50,
            'target_profit': 1500,
            'progress_to_target': 36.7,
        },

        # 5. INVENTORY ALERTS
        'inventory_alerts': {
            'critical': [
                {'item': 'Flour', 'quantity': 0, 'reorder_point': 20, 'action': 'Reorder NOW'},
            ],
            'low': [
                {'item': 'Sugar', 'quantity': 8, 'reorder_point': 10, 'action': 'Reorder soon'},
            ],
            'shopping_list': {
                'Costco': ['Flour (50 lbs)', 'Sugar (20 lbs)'],
                'estimated_cost': 45,
            },
        },

        # 6. PRODUCTION RECOMMENDATIONS
        'production_recommendations': [
            {
                'product': 'Bread Loaf',
                'recommended_quantity': 12,
                'reason': '10 pending orders + 20% buffer',
                'expected_revenue': 72,
                'expected_waste': 0,
            },
        ],

        # 7. TIME ANALYSIS
        'time_analysis': {
            'hours_worked_today': 6.5,
            'hours_worked_this_week': 32,
            'avg_efficiency': 92,  # actual / estimated
            'time_overruns': [
                {'task': 'Research new recipe', 'overrun': 1.5, 'reason': 'Multiple iterations'},
            ],
        },

        # 8. SEASONAL CONTEXT
        'seasonal_context': {
            'current_focus': 'Biochar & Compost Kits',
            'seasonal_tips': [
                'Prepare for holiday season (increase cookie production)',
                'Stock up on winter ingredients (cinnamon, nutmeg)',
            ],
        },

        # 9. INTELLIGENT INSIGHTS
        'intelligent_insights': [
            '📈 Packaging efficiency improved 25% this week (SOP adoption)',
            '💰 "Bread Loaf" most profitable ($4 profit, 2 hrs → $12/hr)',
            '⚠️ "Custom Cake" time overrun 50% (4.5 hrs vs. 3 hrs SOP)',
            '✅ Zero waste last 3 days (production optimization working)',
        ],

        # 10. ACTIONABLE RECOMMENDATIONS
        'recommendations': [
            'Increase "Bread Loaf" production (high profit, low waste)',
            'Create SOP for "Custom Cake" (consistent overruns)',
            'Reduce "Cookies" production (30% waste rate)',
            'Negotiate bulk discount with Costco (monthly spend: $500)',
        ],
    }
```

**Visualization:**

```
=== DAILY BRIEF: November 21, 2025 ===

📊 EXECUTIVE SUMMARY
Revenue: $150  Profit: $45  Hours: 6.5  Tasks: 5  Orders: 3
✅ Exceeded revenue target by 15%
⚠️ Labor cost higher than planned (30% vs. 25%)
🚨 CRITICAL: Flour out of stock

🎯 TODAY'S PRIORITIES
1. [IMMEDIATE] Reorder flour - Blocking 3 orders ($75)
2. [HIGH] Fulfill ORD-002 - Due tomorrow ($30)
3. [HIGH] Package ORD-001 - Overdue by 1 day

💰 REVENUE PIPELINE
Pending: 5 orders ($185)  |  Due This Week: 3  |  Overdue: 2
Monthly Forecast: $2,500 (on track)

📈 FINANCIAL SNAPSHOT
MTD Revenue: $1,200  |  Costs: $650  |  Profit: $550 (45.8%)
Hourly Profit: $12.50  |  Target: $1,500 (37% complete)

🔔 INVENTORY ALERTS
🚨 CRITICAL: Flour (0 lbs) - Reorder NOW
⚠️  LOW: Sugar (8 lbs) - Reorder soon

Shopping List (Costco):
- Flour (50 lbs)
- Sugar (20 lbs)
Estimated Cost: $45

🏭 PRODUCTION RECOMMENDATIONS
- Bread Loaf: 12 units ($72 revenue, 0% waste)

⏱️  TIME ANALYSIS
Today: 6.5 hrs  |  This Week: 32 hrs  |  Efficiency: 92%
Overrun: Research new recipe (+1.5 hrs)

🍂 SEASONAL FOCUS
November: Biochar & Compost Kits
Tips: Prepare for holiday season, stock winter ingredients

🧠 INTELLIGENT INSIGHTS
📈 Packaging efficiency +25% (SOP adoption)
💰 Bread Loaf most profitable ($12/hr)
⚠️ Custom Cake time overrun 50%
✅ Zero waste last 3 days

💡 RECOMMENDATIONS
1. Increase Bread Loaf production (high profit, low waste)
2. Create Custom Cake SOP (consistent overruns)
3. Reduce Cookies production (30% waste)
4. Negotiate Costco bulk discount ($500/mo spend)
```

---

## 3. PRIORITIZED ROADMAP

### Phase 1: Core Tracking (Weeks 1-2) ⚡ **QUICK WINS**

**Goal:** Time & cost tracking operational

**Deliverables:**
- ✅ TimeTrackingParser (Parser #6)
- ✅ CostTrackingParser (Parser #7)
- ✅ Basic profit analysis (Intelligence #1)
- ✅ Updated daily brief with time/cost

**Success Metrics:**
- Track 100% of daily tasks with time/cost
- Calculate true hourly profit
- Detect time overruns automatically

**Timeline:** 2 weeks
**Complexity:** Low (follows existing parser patterns)

---

### Phase 2: Orders & Intelligence (Weeks 3-4) 🚀 **MAJOR FEATURES**

**Goal:** Revenue pipeline + cross-parser insights

**Deliverables:**
- ✅ CustomerOrdersParser (Parser #8)
- ✅ SOPParser (Parser #9)
- ✅ ProductionLogParser (Parser #10)
- ✅ Intelligence Layer (all 5 features)
- ✅ Impressive daily brief (full version)

**Success Metrics:**
- Track 100% of customer orders
- Predict production needs with 95% accuracy
- Reduce waste by 20% via optimization
- Daily brief impresses stakeholders

**Timeline:** 2 weeks
**Complexity:** Medium (new intelligence layer)

---

### Phase 3: Suppliers & Performance (Weeks 5-6) 🔧 **SCALING**

**Goal:** Scale to 10K+ records, complete ecosystem

**Deliverables:**
- ✅ ExpensesParser (Parser #11)
- ✅ SuppliersParser (Parser #12)
- ✅ All 5 performance optimizations
- ✅ Predictive analytics (Intelligence #5)

**Success Metrics:**
- parse_all() < 100ms with 10K records
- Export incremental < 50ms
- Predict revenue/costs with 90% accuracy

**Timeline:** 2 weeks
**Complexity:** Medium-High (performance critical)

---

## 4. IMPLEMENTATION GUIDANCE

### File Structure (After Expansion)

```
elle/coz/
├── __init__.py                    # Updated exports
├── sync_manager.py                # Enhanced with intelligence
├── intelligence.py                # NEW: Intelligence engine
├── daily_brief.py                 # NEW: Daily brief generator
│
├── Existing Parsers (5)
├── kanban_parser.py
├── financials_parser.py
├── schedule_parser.py
├── research_parser.py
├── inventory_parser.py
│
├── New Parsers (7)
├── time_tracking_parser.py        # NEW: Parser #6
├── cost_tracking_parser.py        # NEW: Parser #7
├── customer_orders_parser.py      # NEW: Parser #8
├── sop_parser.py                  # NEW: Parser #9
├── production_log_parser.py       # NEW: Parser #10
├── expenses_parser.py             # NEW: Parser #11
├── suppliers_parser.py            # NEW: Parser #12
│
├── Documentation
├── README.md                       # Updated
├── QUICK_START.md                 # Updated
├── EXPANSION_PLAN.md              # This file
└── IMPLEMENTATION_SUMMARY.md      # Generated post-build
```

**Total:** 19 files (was 6), ~10K lines (was 1.8K)

---

### API Additions (Backward Compatible)

**SyncManager (Enhanced):**
```python
# NEW METHODS (backward compatible - existing code unaffected)
sync.parse_time_tracking()
sync.parse_cost_tracking()
sync.parse_customer_orders()
sync.parse_sops()
sync.parse_production_log()
sync.parse_expenses()
sync.parse_suppliers()

sync.get_profit_analysis()  # NEW: Intelligence #1
sync.get_production_optimization()  # NEW: Intelligence #2
sync.get_task_priorities()  # NEW: Intelligence #3
sync.get_waste_detection()  # NEW: Intelligence #4
sync.get_predictions()  # NEW: Intelligence #5

sync.get_impressive_daily_brief()  # NEW: Full daily brief
```

**Intelligence Engine (New):**
```python
from elle.coz import IntelligenceEngine

intelligence = IntelligenceEngine(sync_manager)

# Analysis
intelligence.analyze_profit()
intelligence.optimize_production()
intelligence.prioritize_tasks()
intelligence.detect_waste()
intelligence.predict_next_30_days()
```

---

### Migration Path

**Phase 1: No breaking changes**
- All new parsers are optional
- Existing code continues working
- New features opt-in

**Phase 2: Gradual adoption**
- Start using time_tracking.csv
- Start using cost_tracking.csv
- Intelligence layer auto-detects available parsers

**Phase 3: Full adoption**
- All 12 parsers operational
- Intelligence layer provides full insights
- Daily brief shows complete picture

---

### Testing Strategy

**Unit Tests (per parser):**
```python
def test_time_tracking_parser():
    parser = TimeTrackingParser('test_data/time_tracking.csv')
    entries = parser.parse()

    assert len(entries) == 10
    assert entries[0].task_id == 'KB-001'
    assert entries[0].variance_hours == 0.5
    assert entries[0].efficiency_score == 0.8
```

**Integration Tests (cross-parser):**
```python
def test_profit_analysis():
    sync = SyncManager()
    sync.parse_all()

    intelligence = IntelligenceEngine(sync)
    profit = intelligence.analyze_profit()

    assert profit['net_profit'] > 0
    assert profit['hourly_profit'] > 0
    assert len(profit['recommendations']) > 0
```

**Performance Tests:**
```python
def test_parse_all_performance():
    sync = SyncManager()

    start = time.time()
    sync.parse_all()
    duration = time.time() - start

    assert duration < 0.1  # <100ms for 10K records
```

**Coverage Target:** 90%+

---

## 5. SUCCESS METRICS

### Performance Targets

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| **Parsers** | 5 | 12 | +140% |
| **Data Points** | ~500 | 10K+ | +2000% |
| **Parse Time** | 50ms | <100ms | 2x data, same speed |
| **Memory Usage** | 5MB | <50MB | 10x data, 10x memory |
| **Export Time** | 20ms | <50ms | Incremental export |
| **Daily Brief Time** | 10ms | <50ms | +Intelligence |

### Quality Targets

| Metric | Target |
|--------|--------|
| **Test Coverage** | 90%+ |
| **Backward Compatibility** | 100% (zero breaking changes) |
| **Documentation** | 100% (all new APIs documented) |
| **Error Handling** | Graceful degradation (missing files OK) |

### Intelligence Targets

| Metric | Target |
|--------|--------|
| **Profit Analysis Accuracy** | 95% |
| **Production Prediction Accuracy** | 90% |
| **Waste Reduction** | 20% via optimization |
| **Time Estimation Accuracy** | 90% (SOP-based) |
| **Revenue Forecast Accuracy** | 85% (30-day) |

### User Experience Targets

| Metric | Target |
|--------|--------|
| **Daily Brief Quality** | "Impressive" rating from stakeholders |
| **Actionable Insights** | 5+ per daily brief |
| **False Positives** | <5% (recommendations) |
| **User Adoption** | 100% (all parsers used daily) |

---

## 6. RISKS & MITIGATION

### Risk #1: Scope Creep

**Risk:** 12 parsers + intelligence = large project
**Mitigation:**
- Phased rollout (3 phases, 2 weeks each)
- MVP for each parser (no premature optimization)
- Reuse existing parser patterns

### Risk #2: Performance Degradation

**Risk:** 10K+ records slow down queries
**Mitigation:**
- Performance optimizations in Phase 3
- Continuous performance testing
- Indexing + caching from day 1

### Risk #3: Data Quality

**Risk:** Garbage in, garbage out (bad CSV data)
**Mitigation:**
- Validation on parse (type checking, range checks)
- Graceful error handling (skip bad rows, log errors)
- Data quality dashboard (detect anomalies)

### Risk #4: Intelligence Accuracy

**Risk:** Predictions/recommendations wrong
**Mitigation:**
- Start with conservative recommendations
- Show confidence scores
- Allow user override
- Track prediction accuracy over time

---

## 7. NEXT STEPS

### Immediate Actions (This Week)

1. **Review & Approve** this expansion plan
2. **Create Coz data files:**
   - `coz/time_tracking.csv`
   - `coz/cost_tracking.csv`
   - `coz/customer_orders.csv`
   - etc. (templates provided)
3. **Set up development environment:**
   - Branch: `feature/coz-expansion`
   - Test data: `coz/test_data/`

### Phase 1 Kickoff (Week 1)

- Implement TimeTrackingParser
- Implement CostTrackingParser
- Basic profit analysis
- Update daily brief
- Write tests

---

## 8. CONCLUSION

This expansion transforms COZ from a **planning file parser** into a **complete operational intelligence system**:

✅ **Time & Cost Tracking** (CORE) - Track actual vs. planned, profit analysis
✅ **Customer Orders** - Revenue pipeline, fulfillment tracking
✅ **SOPs** - Process templates, time/cost estimation
✅ **Production Log** - Track output, waste, actual revenue
✅ **Expenses & Suppliers** - Complete financial picture
✅ **Intelligence Layer** - Cross-parser insights, predictions, recommendations
✅ **Impressive Daily Brief** - Executive dashboard with actionable insights

**Timeline:** 6-8 weeks (3 phases)
**Backward Compatibility:** 100%
**Dependencies:** Zero (pure Python)
**Impact:** 10x more valuable, same simplicity

**Ready to build?** Let's start with Phase 1! 🚀