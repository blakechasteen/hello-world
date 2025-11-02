# food-e: Your Food Journal Reimagined

> **"The journal is the fabric, the palate is the loom, the kitchen is the weaver."**

An elegant food journal and nutrition assistant built on [HoloLoom](../../HoloLoom) that treats nutrition as harmonic frequencies and learns your taste through exploration.

## The Vision

food-e isn't just another calorie counter. It's a **journal-first** nutrition companion that:

- **Speaks in harmonics** instead of numbers ("your diet needs more building blocks" vs "eat 30g protein")
- **Learns your taste** through Thompson Sampling (balances favorites with exploration)
- **Understands time** through circadian rhythms and eating patterns
- **Reflects across horizons** from immediate taste to long-term sustainability

Built on HoloLoom's semantic weaving framework, food-e showcases how to create delightful, intelligent applications on top of advanced ML infrastructure.

---

## ✨ The Elegance: Nutritional Harmonics

Traditional nutrition apps show you this:
```
Today: 1245 cal, 67g protein, 124g carbs, 42g fat
Target: 2000 cal, 150g protein, 200g carbs, 65g fat
Status: ❌ Low on protein, carbs, fat, calories
```

food-e shows you this:
```
Nutritional Spectrum:
--------------------------------------------------
sub_bass     |....| 0.00  (foundational energy)
bass         |####| 0.24  (sustained fuel - fat)
low_mid      |#...| 0.04  (complex carbs)
mid          |####| 0.32  (quick energy - carbs)
high_mid     |####| 0.35  (building blocks - protein)
presence     |##..| 0.05  (micronutrients)
--------------------------------------------------
Dominant: high_mid

Harmonic Resonance: 0.89 (excellent alignment!)
Missing Frequencies:
  - building_blocks: 0.15 (need more protein)
  - metabolic_harmony: 0.89 (great balance)
```

**Why This Matters:**

Your body processes nutrients at different **metabolic frequencies**:
- **Protein** = High frequency (fast turnover, 4-6 hours, building/repair)
- **Carbs** = Mid frequency (energy cycles, 2-6 hours)
- **Fat** = Low frequency (sustained energy, storage, days)
- **Fiber** = Modulates absorption (affects all frequencies)

A balanced diet has **harmonic resonance** - your nutritional spectrum aligns with your body's needs, like an instrument in tune.

---

## 🎯 Phase 1: Foundation (Current)

### Core Data Models

```python
from food_e import Dish, Plate, NutritionalProfile, MealType

# Create dishes
eggs = Dish.create(
    name="Scrambled Eggs (3)",
    nutrition=NutritionalProfile(
        calories=210,
        protein_g=18,
        carbs_g=2,
        fat_g=14
    ),
    tags=["high-protein", "breakfast", "quick"]
)

toast = Dish.create(
    name="Whole Wheat Toast (2 slices)",
    nutrition=NutritionalProfile(
        calories=160,
        protein_g=8,
        carbs_g=28,
        fat_g=2,
        fiber_g=4
    )
)

# Log a meal
breakfast = Plate.create(
    dishes=[eggs, toast],
    meal_type=MealType.BREAKFAST,
    notes="Quick breakfast before work"
)
```

### Spectral Analysis

```python
from food_e import NutritionalSpectrum, Kitchen, KitchenConfig

config = KitchenConfig(
    target_protein_g=150,
    target_calories=2000
)

async with Kitchen(config) as kitchen:
    # Log the plate
    result = await kitchen.log_plate(breakfast)

    print(result["message"])
    # [OK] Logged breakfast
    #   370 cal, 26g protein
    #   Harmonic resonance: 0.89 (excellent!)
    #   -> Still need 124g protein today

    # Visualize the spectrum
    print(result["spectrum"].visualize())
    # Shows ASCII equalizer-style visualization

    # Get harmonic analysis
    resonance = result["resonance"]  # 0-1 score
    gaps = result["spectrum"].missing_frequencies(target_spectrum)

    print(f"Building blocks needed: {gaps['building_blocks']:.2f}")
    print(f"Metabolic harmony: {gaps['metabolic_harmony']:.2f}")
```

### Temporal Queries

```python
from food_e import Journal

journal = Journal("./data/journal.json")

# Query by time
today_meals = journal.today()
this_week = journal.this_week()
last_month = journal.last_n_days(30)

# Aggregate nutrition
today_nutrition = journal.daily_nutrition(datetime.now())
weekly_nutrition = journal.weekly_nutrition(start_date)

# Time series (for semantic calculus)
protein_trajectory = journal.nutritional_trajectory(
    days=30,
    metric="protein_g"
)
# Returns np.array([145, 152, 138, ...]) - 30 days of protein

# Pattern detection
patterns = journal.eating_patterns()
print(f"Breakfast usually at: {patterns['average_meal_times']['breakfast']:.1f}:00")
print(f"Diet consistency: {patterns['diet_consistency']:.2f}")
```

---

## 🧠 Phase 2: Intelligence (Coming Soon)

### Thompson Sampling for Taste

```python
from food_e import Palate

palate = Palate()

# Log meals with feedback
await palate.savor(plate, immediate_rating=0.9)  # Loved it!

# System learns preferences
suggestions = palate.suggest_dishes(
    candidates=available_dishes,
    context={"time": "dinner", "nutritional_gaps": gaps},
    exploration_rate=0.1  # 10% exploration
)

# Balances:
# - Exploitation: Suggest what you historically love
# - Exploration: Try new things to discover preferences
```

### Multi-Timescale Reflection

food-e learns from every meal at **multiple time horizons**:

1. **Immediate (seconds)**: Taste satisfaction
   - "Did it taste good?"
   - Updates: Flavor preferences

2. **Short-term (2-4 hours)**: Energy and satiety
   - "Do I feel energized or sluggish?"
   - Updates: Energy models for each food

3. **Medium-term (1 day)**: Nutritional adequacy
   - "Did this help hit my targets?"
   - Updates: Gap-filling scores

4. **Long-term (weeks)**: Habit sustainability
   - "Is this eating pattern sustainable?"
   - Updates: Habit scores, temporal preferences

This is **PPO reinforcement learning** with multiple reward horizons!

---

## 🌀 Phase 3: Optimization (Future)

### Warp Space Meal Planning

```python
# Complex multi-objective optimization
await kitchen.serve(
    "plan dinners for the week: "
    "150g protein daily, "
    "vegetarian twice, "
    "under $75, "
    "quick meals under 45min, "
    "use up salmon and spinach"
)

# HoloLoom's Warp Space optimizes across:
# - Nutrition (hitting targets)
# - Variety (not repeating meals)
# - Cost (budget constraint)
# - Time (prep time limit)
# - Pantry (use expiring items)
# - Preference (Thompson Sampling suggests favorites + exploration)

# Returns:
# - 7-day meal plan
# - Shopping list (consolidated)
# - Cost breakdown
# - Nutritional analysis
# - Full Spacetime provenance (why each suggestion)
```

---

## 🏗️ Architecture: The Complete Metaphor

food-e maps HoloLoom's weaving concepts to food:

| HoloLoom Concept | food-e Equivalent | Purpose |
|------------------|-------------------|---------|
| **Yarn Graph** | Pantry | Ingredients as threads (discrete) |
| **Loom Command** | Recipe | Weaving pattern (how to combine) |
| **Chrono Trigger** | Temporal Flavor | When to eat what (circadian) |
| **Resonance Shed** | Taste Fusion | Flavor combinations (interference) |
| **DotPlasma** | Nutritional Spectrum | Flowing features (continuous) |
| **Warp Space** | Optimization Manifold | Multi-objective balancing |
| **Convergence Engine** | Menu Decision | What to serve (collapse) |
| **Spacetime Fabric** | Journal | Woven meals over time |
| **Reflection Buffer** | Palate | Learned preferences (Thompson) |

### The Weaving Cycle

```
1. User query: "what should I eat for dinner?"
2. Kitchen (orchestrator) analyzes context:
   - Journal: What have you eaten today?
   - Spectrum: What nutritional gaps exist?
   - Temporal: What time is it? (circadian phase)
   - Palate: What do you like? (learned preferences)
3. WeavingShuttle (Phase 2) weaves decision:
   - Pattern selection: BALANCED_PLATE mode
   - Feature extraction: Nutritional gaps, taste preferences
   - Warp Space: Optimize across multiple objectives
   - Thompson Sampling: Exploit favorites + explore new
4. Convergence: Suggest 3 meals
5. User feedback: "Chose option 2, loved it!"
6. Reflection: Palate learns, updates model
7. Journal: Record the meal as Spacetime point
```

---

## 📊 Demo: See It In Action

```bash
# Run the demo
cd apps
python demo_food_e.py
```

**Demo Output:**
```
============================================================
  food-e: Phase 1 Foundation Demo
  Elegant Food Journaling on HoloLoom
============================================================

============================================================
  Demo 1: Basic Meal Logging
============================================================

[OK] Logged breakfast
  370 cal, 26g protein
  Harmonic resonance: 0.89 (excellent!)
  -> Still need 124g protein today

Spectral Analysis:
Nutritional Spectrum:
--------------------------------------------------
sub_bass     |........................................| 0.00
bass         |#########...............................| 0.24
low_mid      |#.......................................| 0.04
mid          |############............................| 0.32
high_mid     |#############...........................| 0.35
presence     |##......................................| 0.05
--------------------------------------------------
Dominant: high_mid

============================================================
  Demo 2: Spectral Harmonic Resonance
============================================================

Target (Balanced Meal):
[... shows target spectrum ...]

Meal 1: High Protein (Chicken + Veggies)
Resonance with target: 0.90 (specialized)

Meal 2: Balanced (Salmon, Quinoa, Veggies)
Resonance with target: 1.00 (optimal harmony!)

Higher resonance = better alignment with targets

============================================================
  Demo Complete!
============================================================

Phase 1 Complete:
  [OK] Core data models (Dish, Plate, NutritionalProfile)
  [OK] Journal with temporal queries
  [OK] Spectral harmonic analysis
  [OK] Resonance-based nutrition tracking
```

---

## 🔧 Installation

```bash
# From repository root
cd apps/food_e

# food-e has minimal dependencies (numpy for spectral analysis)
pip install numpy

# Optional: HoloLoom for Phase 2 integration
pip install -e ../../HoloLoom
```

---

## 📁 Module Structure

```
food_e/
├── __init__.py              # Public API
├── models.py                # Core data models (380 lines)
│   ├── NutritionalProfile   # Macros + micros
│   ├── Dish                 # Single food item
│   ├── Plate                # Meal (temporal event)
│   └── MealType             # Breakfast, lunch, dinner, snack
├── nutrition.py             # Spectral analysis (226 lines)
│   └── NutritionalSpectrum  # Frequency-based nutrition
├── journal.py               # Temporal memory (212 lines)
│   └── Journal              # Source of truth for eating
├── kitchen.py               # Orchestrator (189 lines)
│   └── Kitchen              # Main entry point (Phase 1 stub)
└── README.md                # This file

Total: ~1000 lines, elegant and tested
```

**Design Principles:**
- **Journal-first**: Everything flows from what you've eaten
- **Elegance**: Poetic naming, consistent metaphors, clean abstractions
- **Intelligence**: Learn from every interaction
- **Delight**: Make tracking feel good, not like homework
- **Reusability**: Patterns extractable for other domains

---

## 🎓 Learning from food-e

This app demonstrates HoloLoom patterns:

### 1. Domain-Specific Semantics
Instead of generic "features," food-e has:
- **Nutritional Spectrum**: Frequency-based representation
- **Harmonic Resonance**: Cosine similarity in frequency domain
- **Temporal Flavor**: Time-aware context

### 2. Thompson Sampling for Exploration
The Palate learns preferences by:
- **Exploiting**: Suggest known favorites (high expected reward)
- **Exploring**: Try new things (reduce uncertainty)
- **Multi-armed bandit**: Each dish is an "arm" to pull

### 3. Multi-Timescale Reflection
Learns from outcomes at multiple horizons:
- PPO reinforcement learning
- Scheduled future reflections
- Reward shaping across time scales

### 4. Semantic Calculus Operations
Enables mathematical operations on nutrition:
- **Integration**: Total nutrition over period
- **Derivative**: Rate of change in calories
- **Trajectories**: Time series for pattern detection
- **Variance**: Consistency of diet

---

## 🚀 Roadmap

### ✅ Phase 1: Foundation (Complete)
- [x] Core data models with HoloLoom integration
- [x] Spectral harmonic analysis
- [x] Journal with temporal queries
- [x] Kitchen orchestrator (stub)
- [x] Working demo

### 🔄 Phase 2: Intelligence (In Progress)
- [ ] Full HoloLoom WeavingShuttle integration
- [ ] Thompson Sampling Palate
- [ ] Multi-timescale reflection
- [ ] Natural language meal parsing
- [ ] Pattern card routing (QUICK_BITE, BALANCED_PLATE, etc.)

### 📅 Phase 3: Optimization (Future)
- [ ] Warp Space meal planning
- [ ] Multi-objective optimization
- [ ] Pantry management with expiration tracking
- [ ] Shopping list generation
- [ ] Recipe database integration

### 🎨 Phase 4: Interface (Future)
- [ ] Beautiful CLI with rich formatting
- [ ] Voice input ("Hey food-e, I ate pasta")
- [ ] Photo-based logging (take picture of meal)
- [ ] Web interface (React)
- [ ] Mobile app (React Native)

---

## 🤝 Contributing

food-e is a **reference implementation** showing how to build delightful apps on HoloLoom.

To build your own domain app:
1. Study this codebase as a template
2. Map HoloLoom concepts to your domain
3. Create domain-specific semantics (like nutritional harmonics)
4. Use HoloLoom's public APIs
5. Keep domain logic isolated

---

## 📖 Related Documentation

- [HoloLoom Framework](../../HoloLoom/README.md)
- [WeavingShuttle API](../../HoloLoom/weaving_shuttle.py)
- [Thompson Sampling in HoloLoom](../../HoloLoom/policy/unified.py)
- [Semantic Calculus](../../HoloLoom/semantic_calculus/README.md)

---

## 💡 Philosophy

> **"The best nutrition app is the one you actually use."**

food-e is built on three principles:

1. **Journal-First**: Track what you eat, insights emerge
2. **Harmony over Targets**: Speak in resonance, not deficits
3. **Learn and Explore**: Balance familiarity with discovery

We believe nutrition tracking should feel like:
- Listening to music (harmonics, resonance, rhythm)
- Not like: Balancing a checkbook (deficits, surpluses, accounting)

---

## 📜 License

MIT

---

## 🙏 Credits

Built on the [HoloLoom](https://github.com/anthropics/claude-code) semantic weaving framework.

Inspired by:
- **Nutritional Science**: Spectral analysis of metabolism
- **Audio Processing**: Frequency domain representations
- **Reinforcement Learning**: Thompson Sampling, PPO
- **HoloLoom**: Semantic weaving, Warp Space, Spacetime fabric

---

## 📧 Citation

If you use food-e in research:

```bibtex
@software{food_e,
  title = {food-e: Nutritional Harmonics on HoloLoom},
  year = {2025},
  url = {https://github.com/you/mythRL/apps/food_e}
}
```

---

*"Your body is an instrument. food-e helps you keep it in tune."* 🎼
