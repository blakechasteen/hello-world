# SOUS – Kitchen Manager MVP
_Handoff for Claude Code Scaffold_

## 0. High-Level Intent

We want to build **SOUS**, a small but extensible "kitchen manager" that can:

1. Take a **menu for an event** (e.g. Thanksgiving).
2. Know the **date of the event** and how many prep days are available.
3. Understand **recipes** with:
   - Make-ahead windows
   - Required equipment (oven, burner, etc.)
   - Duration and dependencies
   - Per-store ingredient preferences
4. Generate:
   - A **multi-store shopping plan** (grouped by store).
   - A **day-by-day prep schedule** that respects:
     - Appliance bottlenecks (limited burners/ovens)
     - Make-ahead rules
     - Optional "if ahead of schedule" tasks
     - Substitutions/decision points.

This scaffold should be **data-first**, with an emphasis on clean models and simple scheduling logic that can be extended later.

The primary use case is the Thanksgiving plan described below.

---

## 1. Core Domain Model

Please model these as Python classes / TypeScript interfaces (your choice) plus JSON-based seed data.

### 1.1 Entities

**Event**
- `id`
- `name` (e.g. "Thanksgiving 2025")
- `date` (e.g. "2025-11-27")
- `prep_days` (e.g. 3–5 days before)
- `appliances` (list of `Appliance` definitions for this kitchen)

**Appliance**
- `id`
- `name` (e.g. "Main Oven", "Burner 1")
- `type` (e.g. `oven`, `burner`, `microwave`)
- `capacity` (for now, 1 = can handle one recipe-task at a time)
- `notes` (e.g. "only one good burner")

**Store**
- `id`
- `name` (e.g. "Amish Market", "Kroger", "Earth Fare", "Asian Market", "Costco")
- `roles` (e.g. ["anchor", "fill", "specialty", "bulk"])
- `priority` (lower number = preferred when multiple stores can supply an ingredient)

**Ingredient**
- `id`
- `name`
- `possible_stores` (list of Store IDs)
- `category` (produce, frozen, dry, plant_dairy, specialty, nuts, alcohol, misc)
- (Optional) `preferred_store` (if strongly tied)

**Recipe**
- `id`
- `name` (e.g. "Herb Stuffing x2", "Mashed Potatoes x2")
- `category` (main, side, dessert, appetizer, drink, optional)
- `ingredients` (list of `RecipeIngredient`)
- `steps` (list of `Step`)
- `make_ahead_min_days` (e.g. 1, 2, 5)
- `make_ahead_max_days` (e.g. 7, or null for "any time before")
- `required_appliances` (subset of `Appliance` types and/or specific IDs)
- `estimated_prep_minutes`
- `estimated_cook_minutes`
- `is_optional` (e.g. muffins = true)
- `substitution_rules` (see below)

**RecipeIngredient**
- `ingredient_id`
- `quantity` (optional for now)
- `unit` (optional; not crucial for MVP)
- `optional` (bool)

**Step**
- `description`
- `appliance_type` (e.g. oven, burner, none)
- `duration_minutes` (estimated)
- `must_be_same_day_as_event` (bool)
- `must_be_day_before` (bool)
- `can_be_any_prep_day` (bool)
- `depends_on` (IDs of other steps in this or other recipes)

**SubstitutionRule**
- `condition` (e.g. "store X didn't have green beans")
- `alternative_ingredient_id`
- `notes`

**DailyPlan**
- `date`
- `tasks` (list of `Task`)

**Task**
- `id`
- `recipe_id`
- `step_id`
- `start_time` (optional for MVP; a rough time window is fine)
- `duration_minutes`
- `appliance_id` (if any)
- `is_optional`
- `status` (planned, completed, skipped)

---

## 2. MVP Features

### 2.1 Input

For now, hard-code the Thanksgiving event + menu as seed data:

- Event date: **Thursday (Thanksgiving)**
- Prep days used: **Monday, Tuesday, Wednesday**

The menu:

- Herb Stuffing ×2
- Mashed Potatoes ×2
- Corn
- Green Beans (Haricots Verts)
- Sweet Potato Casserole
- Cranberry Sauce
- Bread / Rolls
- Pumpkin Pie
- Eggplant Rotel Dip
- Optional: Cranberry Pumpkin Muffins
- Drinks: Cherry Soda, Mulled Apple Cider

Kitchen constraints:

- 1 good burner
- 1 main oven
- Possibly microwave

Stores:

- Amish Market (anchor, cheap heavy produce, bread, cornmeal)
- Vuck Farms (fresh greens & herbs)
- Earth Fare (plant-based dairy, truffle oil, specialty)
- Kroger (fill-in, frozen, packaged)
- Asian Market (shio koji, miso)
- Costco (drinks & bulk)

### 2.2 Outputs

1. **Shopping List by Store**
   - Aggregate all ingredients across recipes.
   - Assign each ingredient to a store based on:
     - `preferred_store` if set
     - or `possible_stores` + store priority
   - Output: `Store -> [Ingredient names]`.

2. **Day-by-Day Prep Schedule**
   - For each recipe, place steps into:
     - **Monday** (2–5 day make-ahead tasks – cranberry sauce)
     - **Tuesday** (bread/cornbread, cubing, drying, optional muffins)
     - **Wednesday** (stuffing assembly, sweet potato casserole base, eggplant Rotel, mashed potatoes)
     - **Thursday** (reheat/finish tasks, green beans, corn, rolls, drinks)
   - Respect:
     - `make_ahead_min_days` / `max_days`
     - `must_be_same_day_as_event`
     - Appliance contention (don't put two oven-heavy tasks in the same narrow window; for MVP, just sequence them).

3. **Optional Tasks**
   - Mark cranberry pumpkin muffins as optional.
   - Only place them if there is a gap on Tuesday, or flag them as "bonus tasks."

---

## 3. Seed Data – Encodings of the Thanksgiving Plan

Use these as rough JSON seed objects. They do not need to be perfect, but they should capture core behavior.

### 3.1 Event

```json
{
  "id": "event_thanksgiving_2025",
  "name": "Thanksgiving 2025",
  "date": "2025-11-27",
  "prep_days": ["2025-11-24", "2025-11-25", "2025-11-26"],
  "appliances": [
    { "id": "oven_main", "name": "Main Oven", "type": "oven", "capacity": 1 },
    { "id": "burner_1", "name": "Front Right Burner", "type": "burner", "capacity": 1 },
    { "id": "microwave", "name": "Microwave", "type": "microwave", "capacity": 1 }
  ]
}
```

### 3.2 Stores

```json
[
  { "id": "store_amish", "name": "Amish Market", "roles": ["anchor"], "priority": 1 },
  { "id": "store_vuck", "name": "Vuck Farms", "roles": ["produce"], "priority": 2 },
  { "id": "store_earthfare", "name": "Earth Fare", "roles": ["specialty", "plant_dairy"], "priority": 3 },
  { "id": "store_kroger", "name": "Kroger", "roles": ["fill"], "priority": 4 },
  { "id": "store_asian", "name": "Asian Market", "roles": ["specialty_asian"], "priority": 5 },
  { "id": "store_costco", "name": "Costco", "roles": ["bulk", "drinks"], "priority": 6 }
]
```

### 3.3 Example Recipe: Cranberry Sauce

```json
{
  "id": "recipe_cranberry_sauce",
  "name": "Cranberry Sauce",
  "category": "side",
  "ingredients": [
    { "ingredient_id": "cranberries" },
    { "ingredient_id": "white_sugar" },
    { "ingredient_id": "orange" },
    { "ingredient_id": "cointreau", "optional": true }
  ],
  "steps": [
    {
      "id": "step_cs_1",
      "description": "Simmer cranberries, sugar, and orange juice/zest until thickened.",
      "appliance_type": "burner",
      "duration_minutes": 20,
      "can_be_any_prep_day": true
    },
    {
      "id": "step_cs_2",
      "description": "Cool and stir in Cointreau.",
      "appliance_type": null,
      "duration_minutes": 5,
      "can_be_any_prep_day": true
    }
  ],
  "make_ahead_min_days": 1,
  "make_ahead_max_days": 5,
  "required_appliances": ["burner"]
}
```

### 3.4 Example Recipe: Herb Stuffing

Focus on make-ahead and oven usage, not exact quantities.

```json
{
  "id": "recipe_stuffing",
  "name": "Herb Stuffing x2",
  "category": "side",
  "ingredients": [
    { "ingredient_id": "cornbread" },
    { "ingredient_id": "onions" },
    { "ingredient_id": "celery" },
    { "ingredient_id": "parsley" },
    { "ingredient_id": "sage" },
    { "ingredient_id": "rosemary" },
    { "ingredient_id": "thyme" },
    { "ingredient_id": "vegan_butter" },
    { "ingredient_id": "bouillon" },
    { "ingredient_id": "white_truffle_oil" }
  ],
  "steps": [
    {
      "id": "step_stuff_bread_prep",
      "description": "Bake or obtain bread/cornbread, cube, dry overnight.",
      "appliance_type": "oven",
      "duration_minutes": 90,
      "can_be_any_prep_day": true
    },
    {
      "id": "step_stuff_assemble",
      "description": "Sauté aromatics, combine with dried bread and bouillon, fold in truffle oil, transfer to baking dish.",
      "appliance_type": "burner",
      "duration_minutes": 60,
      "must_be_day_before": true
    },
    {
      "id": "step_stuff_bake",
      "description": "Bake stuffing until golden.",
      "appliance_type": "oven",
      "duration_minutes": 60,
      "must_be_same_day_as_event": true
    }
  ],
  "make_ahead_min_days": 1,
  "make_ahead_max_days": 2,
  "required_appliances": ["oven", "burner"]
}
```

### 3.5 Example Recipe: Mashed Potatoes (Make-Ahead Variant)

```json
{
  "id": "recipe_mashed_potatoes",
  "name": "Mashed Potatoes x2",
  "category": "side",
  "ingredients": [
    { "ingredient_id": "potatoes" },
    { "ingredient_id": "vegan_butter" },
    { "ingredient_id": "vegan_sour_cream" },
    { "ingredient_id": "oat_milk" },
    { "ingredient_id": "white_truffle_oil" },
    { "ingredient_id": "shio_koji" },
    { "ingredient_id": "salt" }
  ],
  "steps": [
    {
      "id": "step_mp_boil_mash",
      "description": "Boil, drain, and mash potatoes with fats and seasonings; transfer to casserole dish.",
      "appliance_type": "burner",
      "duration_minutes": 60,
      "must_be_day_before": true
    },
    {
      "id": "step_mp_reheat",
      "description": "Reheat mashed potatoes in oven, covered, on low heat.",
      "appliance_type": "oven",
      "duration_minutes": 60,
      "must_be_same_day_as_event": true
    }
  ],
  "make_ahead_min_days": 1,
  "make_ahead_max_days": 1,
  "required_appliances": ["burner", "oven"]
}
```

You can encode the rest of the recipes similarly.

---

## 4. Scheduling Logic (MVP)

Create a small scheduling service/module, e.g. `scheduler.py` or `services/scheduler.ts` with:

```pseudo
generate_daily_plans(event, recipes) -> [DailyPlan]
```

Steps:

1. For each recipe:

   * For each step:

     * Determine eligible days based on:

       * event date
       * `make_ahead_min_days` / `max_days`
       * flags like `must_be_day_before` / `must_be_same_day_as_event` / `can_be_any_prep_day`.

2. Assign steps to days:

   * Monday: longest make-ahead, low-intensity tasks (cranberry sauce, some bread tasks).
   * Tuesday: bread baking/cubing, optional muffins.
   * Wednesday: main assembly tasks (stuffing assemble, casserole base, eggplant mix, mashed potatoes).
   * Thursday: reheat, bake, sauté, drinks.

3. Handle appliance contension:

   * For MVP, just **sequence** oven tasks on the same day (no overlapping).
   * Same for burner tasks: do not double-book the single burner.

4. Mark optional tasks:

   * E.g. cranberry pumpkin muffins as `is_optional = true`.
   * Either:

     * Only include them if day has < X total minutes of planned work, or
     * Include them in a separate "Bonus Tasks if Ahead" list.

---

## 5. Shopping List Logic (MVP)

Create `shopping.py` or `services/shopping.ts` with:

```pseudo
generate_shopping_lists(recipes, ingredients, stores) -> { store_id: [ingredient_ids] }
```

Algorithm:

1. Aggregate all `ingredient_ids` from selected recipes.
2. For each ingredient:

   * If `preferred_store` is set, use that store.
   * Else, choose from `possible_stores` based on store `priority`.
3. Group by `store_id`.
4. Return a dict/map where each key is a store and the value is a deduped list of ingredient names.

---

## 6. Project Structure Suggestion

You can choose exact naming, but something like:

```text
sous/
  app.py                  # CLI or simple HTTP interface (future)
  models/
    __init__.py
    event.py
    recipe.py
    store.py
    schedule.py
  services/
    scheduler.py
    shopping.py
  data/
    ingredients.json
    stores.json
    recipes_thanksgiving.json
    event_thanksgiving.json
  tests/
    test_scheduler.py
    test_shopping.py
```

---

## 7. What to Implement First

1. **Models**: Event, Store, Ingredient, Recipe, Step, DailyPlan, Task.
2. **Seed Data**: Thanksgiving event, stores, ingredients, ~5 core recipes as above.
3. **Shopping List Generator**.
4. **Day-by-Day Scheduler** that:

   * Places steps on Monday–Thursday.
   * Ensures oven tasks are serialized, not parallel, for the single oven.
5. A simple **CLI entrypoint**, e.g.:

```bash
python app.py plan-thanksgiving
```

That prints:

* Shopping lists by store
* Monday–Thursday tasks grouped by day

This is enough to create a working MVP scaffold that we can extend later.
