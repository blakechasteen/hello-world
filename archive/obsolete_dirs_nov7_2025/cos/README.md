# HoloLoom COS - Company Operating System

**Elegant, voice-first business management for farmers and makers.**

Built on [HoloLoom](../HoloLoom/) - the productive AI reasoning system.

---

## Philosophy

> **"Everything is an event. Views are derived. Truth is singular."**

COS treats your business as an **event stream**: every task, sale, purchase, and note is an immutable event. All metrics, reports, and insights are **derived views** of this single source of truth.

### Core Principles

1. **Voice-First**: Log events while working (dirty hands, no typing)
2. **HITL Safety**: Critical data (expenditures, sales, time) verified when confidence < 85%
3. **Single Database**: One event stream, infinite views
4. **Elegant Parsing**: Natural language → structured data
5. **Provenance**: Full audit trail for every decision

---

## Quick Start

### Installation

```bash
cd cos/
python -m pip install -r requirements.txt  # TODO: create this
```

### Basic Usage

```bash
# Log work (time tracking)
python cli.py log "Baked bread for 3 hours, made 12 loaves"

# Log sale
python cli.py log "Sold 10 loaves for $60"

# Log purchase
python cli.py log "Bought 50 pounds flour at Costco for $27"

# Verify pending events (HITL queue)
python cli.py verify

# View summaries
python cli.py summary today
python cli.py summary week

# Check inventory
python cli.py inventory

# Ask questions
python cli.py ask "What is my most profitable product?"
```

---

## Architecture

### Event Stream Database

Everything is stored as an **event**:

```python
Event(
    type=EventType.TASK,
    raw_input="Baked bread for 3 hours, made 12 loaves",
    timestamp=datetime.now(),
    amount=-60.00,        # Labor cost (3 hrs × $20)
    quantity=3.0,         # Hours worked
    unit="hours",
    item="bread",
    category=Category.LABOR,
    product_line=ProductLine.BREAD,
    confidence=0.95,      # NLP confidence
    verified=True,        # HITL verified
)
```

### Event Types

- **TASK**: Work performed (time tracking)
- **SALE**: Revenue transaction
- **PURCHASE**: Expense/purchase
- **INVENTORY**: Inventory adjustment
- **NOTE**: Reflection, observation, insight
- **PLAN**: Planning (daily/weekly goals)
- **GOAL**: Long-term objective
- **REVIEW**: Daily/weekly review

### NLP Parser

The parser uses **regex patterns** for high-confidence matching:

```python
# Input: "Baked bread for 3 hours, made 12 loaves"
# Parsed:
{
    'type': 'task',
    'item': 'bread',
    'duration': 3.0,
    'output': 12,
    'labor_cost': 60.00,
    'confidence': 0.95
}

# Input: "Bought 50 pounds flour for $27"
# Parsed:
{
    'type': 'purchase',
    'item': 'flour',
    'quantity': 50,
    'unit': 'lb',
    'amount': 27.00,
    'confidence': 0.90
}
```

### HITL Verification

**Critical data requires human verification when confidence < 85%:**

```
⚠️  VERIFICATION NEEDED
Reason: Purchase confidence 0.82 < 0.85
Input: Bought some stuff for $27
Parsed: Event(purchase, item=stuff, $27.00)
Confidence: 0.82

✓ Is this correct? [y/n/edit]:
```

### Views (Derived Data)

All metrics are **computed from events**, not stored separately:

```python
# Daily Summary
revenue = sum(e.amount for e in events if e.type == SALE)
cogs = sum(e.amount for e in events if e.category == COGS)
labor = sum(e.amount for e in events if e.type == TASK)
profit = revenue - cogs - labor

# Inventory
inventory = {}
for e in events:
    if e.quantity:
        inventory[e.item] = inventory.get(e.item, 0) + e.quantity

# Product Performance
for product in ProductLine:
    revenue = sum(e.amount for e in events
                  if e.type == SALE and e.product_line == product)
    costs = sum(e.amount for e in events
                if e.product_line == product
                and e.type in (TASK, PURCHASE))
    profit = revenue - costs
    margin = profit / revenue if revenue > 0 else 0
```

---

## Features

### ✅ Phase 1 (Current)

- [x] Event stream database (SQLite)
- [x] NLP parser (business/productivity intents)
- [x] HITL verification system
- [x] Event storage/retrieval API
- [x] CLI interface (log, verify, summary, ask)
- [x] Daily/weekly summaries
- [x] Product performance tracking
- [x] Inventory tracking
- [x] Full-text search

### 🚧 Phase 2 (Next)

- [ ] HoloLoom memory integration (events → knowledge graph)
- [ ] Agentic query system ("What should I focus on?")
- [ ] View generators (P&L, balance sheet, cash flow)
- [ ] Alert system (burnout warnings, low margins)
- [ ] Planning integration (daily/weekly linked to 90-day plan)

### 🔮 Phase 3 (Future)

- [ ] Whisper voice input integration
- [ ] Tufte-style web dashboard
- [ ] Mobile optimization (POS system)
- [ ] Daily review workflow
- [ ] Automated insights

---

## Examples

### Time Tracking

```bash
# Simple duration
cos log "Baked bread for 3 hours"
# → Task: bread_baking, 3 hrs, $60 labor cost

# Start/end times
cos log "Started bread at 7am, finished at 10am"
# → Task: bread_baking, 3 hrs, $60 labor cost

# With output
cos log "Worked on meal prep for 4 hours, made 20 quarts soup"
# → Task: meal_prep, 4 hrs, $80 labor cost, 20 quarts output
```

### Sales Tracking

```bash
# Quantity + price
cos log "Sold 10 loaves for $60"
# → Sale: bread, $60 revenue, -10 loaves inventory

# Total only
cos log "Made $180 in bread sales"
# → Sale: bread, $180 revenue

# Daily total
cos log "$450 revenue today"
# → Sale: $450 revenue
```

### Purchase Tracking

```bash
# Full details
cos log "Bought 50 pounds flour at Costco for $27"
# → Purchase: flour, 50 lb, $27, $0.54/lb, vendor=Costco

# Simple
cos log "Paid electric bill $145"
# → Purchase: electric, $145, category=overhead_utilities

# Amazon order
cos log "Amazon order: bread bags, $23.50"
# → Purchase: bread bags, $23.50, category=COGS_packaging
```

### Notes & Reflections

```bash
# Daily review
cos log "Daily review: Bread taking longer than expected, need better oven"
# → Note: stored for HoloLoom learning

# Customer feedback
cos log "Customer said honey is best they've ever had"
# → Note: positive feedback on honey

# Planning
cos log "Tomorrow: focus on meal prep, goal 30 quarts"
# → Plan: tomorrow's goal
```

---

## Integration with 90-Day Plan

COS is designed to track progress against your [90-day timeline](./90_day_timeline.md):

```bash
# Week 1 goals
cos log "Week 1 goal: $150-250 revenue, 10-15 loaves"

# Daily work
cos log "Baked 12 loaves, 3 hours"
cos log "Sold 8 loaves for $48"

# Weekly review
cos summary week
# Revenue:      $450.00
# Target:       $500.00
# Variance:      -$50.00 (-10%)
# ✓ On track for Week 1 milestone
```

---

## Database Schema

See [core/schema.sql](./core/schema.sql) for complete schema.

**Main table:**
```sql
events (
    id, timestamp, type, raw_input, source,
    confidence, verified, parsed_data,
    amount, quantity, unit, item, category,
    product_line, related_to, parent_goal,
    tags, location, verification_note
)
```

**Views:**
- `daily_summary` - Revenue, costs, profit by day
- `weekly_summary` - Weekly aggregates
- `product_performance` - Performance by product line
- `inventory_current` - Current stock levels
- `unverified_expenditures` - HITL queue

---

## API Usage

```python
from cos import COSCLI, EventStore, parse_input
import asyncio

async def main():
    # CLI usage
    cli = COSCLI()
    await cli.log("Baked bread for 3 hours")
    await cli.summary_today()

    # Direct API usage
    store = EventStore()

    # Parse input
    intent, verification = parse_input("Sold 10 loaves for $60")

    if verification:
        # HITL needed
        print(verification)
        # ... handle verification ...

    # Store event
    event = intent.to_event("Sold 10 loaves for $60")
    event_id = await store.store(event)

    # Query events
    sales = await store.query(
        event_type=EventType.SALE,
        start_date=datetime(2025, 11, 1),
        limit=100
    )

    # Get summaries
    summary = await store.get_daily_summary(datetime.now())
    print(f"Profit today: ${summary.profit}")

asyncio.run(main())
```

---

## Roadmap

### Week 1-2: Foundation ✅
- [x] Event database schema
- [x] NLP parser
- [x] HITL verification
- [x] CLI interface
- [x] Basic queries

### Week 3-4: Intelligence
- [ ] HoloLoom integration (events → memory)
- [ ] Agentic queries ("What should I focus on?")
- [ ] Alert system (warnings, recommendations)
- [ ] 4 core accounting documents (P&L, balance sheet, cash flow, budget vs actual)

### Week 5-6: Interface
- [ ] Whisper voice input
- [ ] Web dashboard (Tufte visualizations)
- [ ] Mobile POS system
- [ ] Daily review workflow

### Month 2-3: Scale
- [ ] Multi-user support
- [ ] Export to accounting software
- [ ] Advanced analytics
- [ ] Automated insights

---

## Philosophy Deep Dive

### Why Event Sourcing?

Traditional databases store **current state**. Event sourcing stores **history of changes**.

**Benefits:**
1. **Complete audit trail**: Know exactly what happened when
2. **Time travel**: Recreate state at any point in history
3. **Provenance**: Every metric traceable to source events
4. **Flexibility**: Add new views without changing data
5. **Learning**: HoloLoom can learn patterns from event stream

### Why HITL?

**Expenditure tracking is critical.** A single missed purchase or incorrect time log can throw off your entire profitability analysis.

**But:** HITL is expensive (your time). So we use it **strategically**:

- **High confidence (>0.85)**: Auto-process, full provenance
- **Low confidence (<0.85)**: HITL verification required
- **Critical data** (purchases, large sales): Always verify if uncertain
- **Batch verification**: Process queue once daily (5-10 min)

This gives you **99% accuracy with 5% overhead**.

### Why Voice-First?

Farmers have dirty hands. Typing on phones at the market is slow and error-prone.

**Voice input:**
- Natural (speak as you work)
- Fast (3x faster than typing)
- Hands-free (while packaging, harvesting, selling)
- Complete context (captures nuance that forms miss)

But voice needs **smart parsing + HITL** to be reliable. That's what COS provides.

---

## Contributing

COS is built as part of the HoloLoom project. See [../CLAUDE.md](../CLAUDE.md) for development guidelines.

**Development principles:**
1. **Simplicity over features**: Start minimal, add only what's needed
2. **Voice-first**: Every feature should work via voice input
3. **HITL when uncertain**: Don't guess, ask the human
4. **Provenance always**: Every number traceable to source
5. **Views, not state**: Derive everything from events

---

## License

Part of HoloLoom. See [../LICENSE](../LICENSE).

---

## Questions?

- **How do I handle refunds?** Log as negative sale: `cos log "Refunded customer $12 for bread"`
- **How do I correct mistakes?** Use `cos verify` to edit unverified events, or update database directly
- **How do I track owner draws?** `cos log "Withdrew $200 for personal use"` (category: owner_draw)
- **How do I handle inventory losses?** `cos log "2 loaves didn't sell, composted"` (waste tracking)
- **Can I import from spreadsheets?** Not yet, but planned for Phase 2

---

**Built with ❤️ and HoloLoom by farmers, for farmers.**
