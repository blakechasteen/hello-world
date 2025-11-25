# SOUS - Kitchen Manager MVP

**Built with HoloLoom Concurrent Metaprompting** 🚀

SOUS is a small but extensible kitchen manager that handles multi-store shopping routing, day-by-day prep scheduling, appliance conflict detection, and make-ahead task planning.

---

## 🎯 What SOUS Does

1. **Multi-Store Shopping Routing** - Automatically routes ingredients to the best stores based on priority
2. **Day-by-Day Prep Scheduling** - Schedules recipe tasks across multiple prep days based on make-ahead windows
3. **Appliance Conflict Detection** - Prevents double-booking of ovens, burners, and other appliances
4. **Make-Ahead Task Planning** - Intelligently places tasks to maximize freshness and minimize stress
5. **Optional Task Management** - Suggests "bonus tasks" when you're ahead of schedule
6. **🆕 Pantry & Fridge Inventory** - Tracks what you already have (pantry, fridge, freezer, counter)
7. **🆕 Expiration Tracking** - Alerts you to items expiring soon or already expired
8. **🆕 Smart Shopping Lists** - Automatically skips items you already have in inventory
9. **🆕 Leftover Management** - Tracks leftovers after meals with expiration dates and consumption status

---

## 🚀 Quick Start

### Run Thanksgiving Planner

```bash
cd Sous
python sous/app.py
```

Or from package:

```bash
cd Sous
python -m sous.app
```

### Run Demo

```bash
python demos/demo_thanksgiving.py
```

### Run Tests

```bash
# Thanksgiving integration tests
python tests/test_thanksgiving.py

# Inventory management tests
python tests/test_inventory.py
```

---

## 🏪 Inventory Management

SOUS now tracks your pantry, fridge, and freezer inventory!

### Features

- **Track Locations**: Pantry, Fridge, Freezer, Counter
- **Expiration Dates**: Automatic alerts for items expiring soon or expired
- **Smart Shopping**: Shopping lists automatically skip items you already have
- **Leftover Tracking**: Track post-meal leftovers with expiration dates
- **Quantity Management**: Track quantities with units (items, lb, oz, cup, etc.)

### Inventory Data Files

- `sous/data/inventory.json` - Current inventory (pantry/fridge/freezer)
- `sous/data/leftovers.json` - Post-meal leftovers with consumption status

### Example Inventory Output

```
================================================================================
🏪 PANTRY & FRIDGE INVENTORY
================================================================================

📦 PANTRY: 6 items
   • Salt: 1.0 package
      Note: Morton iodized salt, half used
   • White Sugar: 2.0 lb
      Note: Domino sugar, plenty left
   • Flour: 3.0 lb
      Note: All-purpose flour

📦 FRIDGE: 2 items
   • Vegan Butter: 2.0 package [Expires in 21 days]
      Note: Earth Balance, 2 sticks remaining
   • Oat Milk: 0.5 package [Expires in 6 days]
      Note: Oatly, half carton left

🛒 Generating shopping lists (accounting for inventory)...
   ✓ Shopping lists generated for 6 stores
   ✓ 11 items skipped (already in inventory!)
```

---

## 📊 Output Example

```
🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃
SOUS - Thanksgiving 2025 Kitchen Manager
🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃🦃

📥 Loading data...
   ✓ Loaded 9 recipes
   ✓ Loaded 6 stores
   ✓ Event: Thanksgiving 2025 on Thursday, November 27, 2025

================================================================================
📋 SHOPPING LISTS BY STORE
================================================================================

🏪 Amish Market
   Priority: 1
   Items (12):
     • Celery
     • Cornbread or Cornmeal
     • Eggplant
     • Fresh Parsley
     • ...

🏪 Earth Fare
   Priority: 3
   Items (8):
     • Oat Milk
     • Vegan Butter
     • Vegan Sour Cream
     • White Truffle Oil
     • ...

================================================================================
📅 PREP SCHEDULE (Day-by-Day)
================================================================================

🟣 MONDAY — FOUNDATION + EARLY PREP
Date: Monday, November 24, 2025
Total Work: 25 minutes (0h 25m)
Tasks: 1

Scheduled Tasks:
   1. Cranberry Sauce
      Simmer cranberries, sugar, and orange juice/zest until thickened (20 min).
      Duration: 20 min (burner_1)

🟡 WEDNESDAY — MAJOR PREP DAY
Date: Wednesday, November 26, 2025
Total Work: 240 minutes (4h 0m)
Tasks: 7

Scheduled Tasks:
   1. Herb Stuffing x2
      Sauté aromatics, combine with dried bread...
      Duration: 60 min (burner_1)

   2. Mashed Potatoes x2
      Boil, drain, and mash potatoes...
      Duration: 60 min (burner_1)
   ...
```

---

## 🏗️ Project Structure

```
Sous/
├── sous/                      # Core application
│   ├── models/                # Data models
│   │   ├── event.py           # Event + Appliance
│   │   ├── recipe.py          # Recipe + Step + Ingredient
│   │   ├── store.py           # Store
│   │   └── schedule.py        # DailyPlan + Task
│   ├── services/              # Business logic
│   │   ├── scheduler.py       # Scheduling engine
│   │   └── shopping.py        # Shopping list generator
│   ├── data/                  # Seed data
│   │   ├── event_thanksgiving.json
│   │   ├── recipes_thanksgiving.json
│   │   ├── stores.json
│   │   └── ingredients.json
│   └── app.py                 # CLI entry point
│
├── metaprompts/               # Metaprompting workspace
│   ├── orchestrator.py        # Concurrent strategy runner
│   ├── original_prompt_*.md   # Original SOUS prompts
│   ├── strategies/            # 10 strategy outputs
│   └── synthesis_guide.md     # How to synthesize
│
├── tests/
│   └── test_thanksgiving.py   # Integration tests
│
├── demos/
│   └── demo_thanksgiving.py   # Demo script
│
├── docs/
│   └── architecture.md        # Architecture details
│
└── README.md                  # This file
```

---

## 🧪 Running Tests

All tests should pass:

```bash
cd Sous
python tests/test_thanksgiving.py
```

**Expected Output:**
```
SOUS Integration Tests - Thanksgiving Planning
✓ Data loading test passed
✓ Shopping list generation test passed (6 stores)
✓ Schedule generation test passed (XX total tasks)
✓ Conflict detection test passed
✓ Multi-store routing test passed

Test Results: 5 passed, 0 failed
```

---

## 🎨 Metaprompting Integration

SOUS was built using **HoloLoom's concurrent metaprompting system**, applying all 10 advanced prompting strategies in parallel:

1. **VERIFY** - Chain of Verification for gap-finding
2. **CHALLENGE** - Adversarial prompting for risk analysis
3. **REVERSE** - Backward reasoning from ideal state
4. **OPTIMIZE** - Recursive refinement
5. **DEEP** - Extreme technical detail
6. **SCAFFOLD** - Structured organization
7. **PRIME** - Comparative analysis
8. **DEBATE** - Multi-perspective design
9. **TEACH** - Educational clarity
10. **TEMP-SIM** - Alternative architectures

### Run Metaprompting (Optional)

```bash
cd Sous/metaprompts
python orchestrator.py
```

This generates 10 enhanced metaprompts (one per strategy) that can be sent to Claude for comprehensive architecture analysis.

---

## 🔧 Key Features

### 1. Smart Store Routing

- Ingredients automatically routed to optimal stores
- Priority-based selection (lower number = higher priority)
- Store roles (anchor, specialty, bulk, produce)
- Preferred store overrides

### 2. Intelligent Scheduling

- Make-ahead window constraints (min/max days)
- Same-day vs. day-before task placement
- Flexible prep day tasks
- Optional task suggestions

### 3. Appliance Conflict Detection

- Tracks oven, burner, microwave usage
- Detects double-booking
- Suggests task reordering

### 4. Substitution Support

- Built-in decision trees
- Alternative ingredient suggestions
- Conditional task placement

---

## 📈 Future Enhancements

**Planned Features:**
- Recipe scaling (2x, 4x, etc.)
- Dietary restriction filtering (vegan, gluten-free, etc.)
- Skill level adaptation (beginner → expert)
- Time window optimization (precise start times)
- Geographic store routing
- Mobile app interface
- Voice assistant integration

See `metaprompts/synthesis_guide.md` for insights from all 10 metaprompting strategies.

---

## 📚 Documentation

- **[README.md](README.md)** - This file (quick start)
- **[docs/architecture.md](docs/architecture.md)** - Detailed architecture
- **[metaprompts/synthesis_guide.md](metaprompts/synthesis_guide.md)** - Metaprompting insights
- **[metaprompts/original_prompt_*.md](metaprompts/)** - Original SOUS analysis

---

## 🎉 What You Accomplished

You didn't just build a Thanksgiving schedule — you **prototyped the core operating model of SOUS**.

This MVP demonstrates:
- ✅ Multi-store shopping orchestration
- ✅ Day-by-day prep scheduling
- ✅ Appliance bottleneck logic
- ✅ Decision trees and substitutions
- ✅ Optional task management
- ✅ Complete data-driven architecture

**All patterns from your manual Thanksgiving planning are now automated!**

---

## 🤝 Contributing

SOUS is extensible by design. To add features:

1. **New Recipes** - Add to `sous/data/recipes_thanksgiving.json`
2. **New Stores** - Add to `sous/data/stores.json`
3. **New Services** - Add to `sous/services/`
4. **New Models** - Add to `sous/models/`

---

## 📄 License

MIT License - Built with HoloLoom

---

**Built with ❤️ using HoloLoom Concurrent Metaprompting**
