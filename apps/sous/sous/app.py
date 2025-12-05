#!/usr/bin/env python3
"""
SOUS - Kitchen Manager CLI

Main entry point for SOUS system
"""

import json
import sys
from pathlib import Path
from datetime import date

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sous.models.event import Event
from sous.models.recipe import Recipe
from sous.models.store import Store
from sous.models.inventory import InventoryItem, Leftover, WastedItem, GardenHarvest
from sous.services.scheduler import Scheduler
from sous.services.shopping import ShoppingListGenerator
from sous.services.inventory_manager import InventoryManager


def load_data():
    """Load all seed data from JSON files"""
    data_dir = Path(__file__).parent / 'data'

    # Load stores
    with open(data_dir / 'stores.json') as f:
        stores_data = json.load(f)
    stores = [Store.from_dict(s) for s in stores_data]

    # Load event
    with open(data_dir / 'event_thanksgiving.json') as f:
        event_data = json.load(f)
    event = Event.from_dict(event_data)

    # Load ingredients
    with open(data_dir / 'ingredients.json') as f:
        ingredients = json.load(f)

    # Load recipes
    with open(data_dir / 'recipes_thanksgiving.json') as f:
        recipes_data = json.load(f)
    recipes = [Recipe.from_dict(r) for r in recipes_data]

    # Load inventory
    inventory_items = []
    inventory_file = data_dir / 'inventory.json'
    if inventory_file.exists():
        with open(inventory_file) as f:
            inventory_data = json.load(f)
        inventory_items = [InventoryItem.from_dict(item) for item in inventory_data]

    # Load leftovers
    leftovers = []
    leftovers_file = data_dir / 'leftovers.json'
    if leftovers_file.exists():
        with open(leftovers_file) as f:
            leftovers_data = json.load(f)
        leftovers = [Leftover.from_dict(lo) for lo in leftovers_data]

    # Load waste log
    waste_log = []
    waste_file = data_dir / 'waste_log.json'
    if waste_file.exists():
        with open(waste_file) as f:
            waste_data = json.load(f)
        waste_log = [WastedItem.from_dict(w) for w in waste_data]

    # Load garden harvests
    garden_harvests = []
    harvest_file = data_dir / 'garden_harvests.json'
    if harvest_file.exists():
        with open(harvest_file) as f:
            harvest_data = json.load(f)
        garden_harvests = [GardenHarvest.from_dict(h) for h in harvest_data]

    # Load seasonal produce calendar
    seasonal_produce = []
    seasonal_file = data_dir / 'seasonal_produce.json'
    if seasonal_file.exists():
        with open(seasonal_file) as f:
            seasonal_produce = json.load(f)

    return event, recipes, stores, ingredients, inventory_items, leftovers, waste_log, garden_harvests, seasonal_produce


def print_shopping_lists(shopping_lists, stores):
    """Pretty-print shopping lists by store"""
    print("\n" + "=" * 80)
    print("📋 SHOPPING LISTS BY STORE")
    print("=" * 80)

    # Get store objects
    store_map = {s.id: s for s in stores}

    # Sort stores by priority
    sorted_store_ids = sorted(
        shopping_lists.keys(),
        key=lambda sid: store_map[sid].priority if sid in store_map else 999
    )

    for store_id in sorted_store_ids:
        items = shopping_lists[store_id]
        if not items:
            continue

        store = store_map.get(store_id)
        if not store:
            continue

        print(f"\n🏪 {store.name}")
        print(f"   Roles: {', '.join(store.roles)}")
        print(f"   Priority: {store.priority}")
        print(f"   Items ({len(items)}):")

        for item in sorted(items):
            print(f"     • {item}")


def print_daily_plans(daily_plans, recipes):
    """Pretty-print day-by-day schedule"""
    print("\n" + "=" * 80)
    print("📅 PREP SCHEDULE (Day-by-Day)")
    print("=" * 80)

    # Get recipe map for lookups
    recipe_map = {r.id: r for r in recipes}

    # Day labels
    day_labels = {
        0: "🟢 THURSDAY — SHOWTIME (Event Day)",
        1: "🟡 WEDNESDAY — MAJOR PREP DAY",
        2: "🟠 TUESDAY — BREAD & DRY GOODS",
        3: "🟣 MONDAY — FOUNDATION + EARLY PREP"
    }

    for plan in daily_plans:
        days_before = (plan.date - daily_plans[0].date).days if daily_plans else 0
        days_before_abs = abs(days_before)

        # Print day header
        label = day_labels.get(days_before_abs, f"DAY {plan.date.isoformat()}")
        print(f"\n{label}")
        print(f"Date: {plan.date.strftime('%A, %B %d, %Y')}")
        print(f"Total Work: {plan.total_duration_minutes()} minutes ({plan.total_duration_minutes() // 60}h {plan.total_duration_minutes() % 60}m)")
        print(f"Tasks: {len(plan.tasks)}")

        if not plan.tasks:
            print("   (No tasks scheduled)")
            continue

        print("\nScheduled Tasks:")
        print("-" * 80)

        for idx, task in enumerate(plan.tasks, 1):
            recipe = recipe_map.get(task.recipe_id)
            recipe_name = recipe.name if recipe else task.recipe_id
            step = recipe.get_step(task.step_id) if recipe else None
            step_desc = step.description if step else "Unknown step"

            optional_tag = " [OPTIONAL]" if task.is_optional else ""
            appliance_tag = f" ({task.appliance_id})" if task.appliance_id else ""

            print(f"\n   {idx}. {recipe_name}{optional_tag}")
            print(f"      {step_desc}")
            print(f"      Duration: {task.duration_minutes} min{appliance_tag}")

    print("\n" + "=" * 80)


def print_conflicts(conflicts):
    """Print appliance conflicts if any"""
    if not conflicts:
        print("\n✅ No appliance conflicts detected!")
        return

    print("\n" + "=" * 80)
    print("⚠️  APPLIANCE CONFLICTS DETECTED")
    print("=" * 80)

    for conflict in conflicts:
        print(f"\n⚠️  {conflict['date']}: {conflict['message']}")
        print(f"   Tasks affected:")
        for task_id in conflict['task_ids']:
            print(f"     • {task_id}")


def print_inventory(inventory_manager, ingredients):
    """Print current inventory status"""
    print("\n" + "=" * 80)
    print("🏪 PANTRY & FRIDGE INVENTORY")
    print("=" * 80)

    from sous.models.inventory import InventoryLocation

    if not inventory_manager.inventory:
        print("\n   (No items in inventory)")
        return

    # Group by location
    for location in InventoryLocation:
        items = inventory_manager.get_items_by_location(location)
        if not items:
            continue

        print(f"\n📦 {location.value.upper()}: {len(items)} items")
        for item in items:
            ing_name = ingredients.get(item.ingredient_id, {}).get("name", item.ingredient_id)
            quantity_str = f"{item.quantity} {item.unit.value}"

            expiration_str = ""
            if item.expiration_date:
                days_left = item.days_until_expiration()
                if days_left is not None:
                    if days_left < 0:
                        expiration_str = f" [EXPIRED {abs(days_left)} days ago]"
                    elif days_left <= 3:
                        expiration_str = f" [Expires in {days_left} days]"

            print(f"   • {ing_name}: {quantity_str}{expiration_str}")
            if item.notes:
                print(f"      Note: {item.notes}")

    # Show expiring soon
    expiring = inventory_manager.get_expiring_soon(days=3)
    if expiring:
        print(f"\n⚠️  {len(expiring)} items expiring within 3 days")

    # Show expired
    expired = inventory_manager.get_expired()
    if expired:
        print(f"\n❌ {len(expired)} expired items (should be removed)")


def print_leftovers(leftovers):
    """Print leftover status"""
    if not leftovers:
        return

    print("\n" + "=" * 80)
    print("🍽️  LEFTOVERS (After Thanksgiving)")
    print("=" * 80)

    active_leftovers = [lo for lo in leftovers if not lo.consumed]
    consumed_leftovers = [lo for lo in leftovers if lo.consumed]

    if active_leftovers:
        print(f"\n🥡 Active Leftovers: {len(active_leftovers)}")
        for lo in active_leftovers:
            days_left = lo.days_remaining()
            status = ""
            if days_left < 0:
                status = f" [EXPIRED {abs(days_left)} days ago]"
            elif days_left <= 1:
                status = f" [Eat TODAY! {days_left} days left]"
            elif days_left <= 2:
                status = f" [Eat SOON! {days_left} days left]"

            print(f"   • {lo.description}: {lo.quantity} {lo.unit.value}{status}")
            if lo.notes:
                print(f"      {lo.notes}")

    if consumed_leftovers:
        print(f"\n✅ Consumed: {len(consumed_leftovers)}")
        for lo in consumed_leftovers:
            print(f"   • {lo.description} ({lo.notes})")


def print_waste_statistics(inventory_manager, ingredients):
    """Print food waste statistics and learnings"""
    if not inventory_manager.waste_log:
        return

    print("\n" + "=" * 80)
    print("🗑️  FOOD WASTE TRACKING & COST ANALYSIS")
    print("=" * 80)

    # Get statistics
    stats = inventory_manager.get_waste_statistics()

    print(f"\n📊 Overall Statistics:")
    print(f"   Total items wasted: {stats['total_items_wasted']}")
    print(f"   Total cost wasted: ${stats['total_cost']:.2f}")
    if stats['avg_shelf_life_days']:
        print(f"   Average shelf life: {stats['avg_shelf_life_days']:.1f} days")

    # Waste by reason
    if stats['by_reason']:
        print(f"\n📋 Waste by Reason:")
        for reason, data in sorted(stats['by_reason'].items(), key=lambda x: x[1]['count'], reverse=True):
            print(f"   • {reason.capitalize()}: {data['count']} items (${data['cost']:.2f})")

    # Most wasted ingredients
    most_wasted = inventory_manager.get_most_wasted_ingredients(limit=3)
    if most_wasted:
        print(f"\n⚠️  Most Frequently Wasted:")
        for ing_id, count, cost in most_wasted:
            ing_name = ingredients.get(ing_id, {}).get("name", ing_id)
            actual_shelf_life = inventory_manager.calculate_actual_shelf_life(ing_id)
            shelf_life_str = f" (avg {actual_shelf_life:.0f} days)" if actual_shelf_life else ""
            print(f"   • {ing_name}: {count}x wasted, ${cost:.2f}{shelf_life_str}")

    # Recent waste (last 30 days)
    recent_stats = inventory_manager.get_waste_statistics(days=30)
    if recent_stats['total_items_wasted'] > 0:
        print(f"\n📅 Last 30 Days:")
        print(f"   Items wasted: {recent_stats['total_items_wasted']}")
        print(f"   Cost: ${recent_stats['total_cost']:.2f}")


def print_garden_harvests(garden_harvests, ingredients):
    """Print garden harvest summary"""
    if not garden_harvests:
        return

    print("\n" + "=" * 80)
    print("🌱 GARDEN HARVESTS")
    print("=" * 80)

    # Calculate total value
    total_value = sum(h.estimated_value for h in garden_harvests if h.estimated_value)

    print(f"\nTotal harvests: {len(garden_harvests)}")
    print(f"Estimated retail value: ${total_value:.2f} (savings from growing your own!)")

    # Group by season
    by_season = {}
    for harvest in garden_harvests:
        season = harvest.season.value
        if season not in by_season:
            by_season[season] = []
        by_season[season].append(harvest)

    for season in ["spring", "summer", "fall", "winter"]:
        harvests = by_season.get(season, [])
        if not harvests:
            continue

        print(f"\n🍃 {season.upper()} HARVESTS ({len(harvests)})")
        for h in harvests:
            ing_name = ingredients.get(h.ingredient_id, {}).get("name", h.ingredient_id)
            value_str = f" (${h.estimated_value:.2f})" if h.estimated_value else ""
            print(f"   • {ing_name}: {h.quantity} {h.unit.value}{value_str}")
            print(f"      {h.garden_location} - {h.harvest_date.strftime('%B %d')}")
            if h.notes:
                print(f"      {h.notes}")


def print_seasonal_calendar(seasonal_produce):
    """Print seasonal produce availability"""
    if not seasonal_produce:
        return

    print("\n" + "=" * 80)
    print("📅 SEASONAL PRODUCE CALENDAR")
    print("=" * 80)

    from datetime import date
    current_month = date.today().month

    # Get what's in season now
    in_season_now = [p for p in seasonal_produce if current_month in p["available_months"]]

    if in_season_now:
        print(f"\n✅ IN SEASON NOW ({len(in_season_now)} items):")
        for produce in in_season_now:
            price_diff = produce["typical_price_low_season"] - produce["typical_price_high_season"]
            savings = (price_diff / produce["typical_price_low_season"]) * 100
            print(f"   • {produce['name']}")
            print(f"      Peak price: ${produce['typical_price_high_season']:.2f} (save {savings:.0f}%)")
            if produce.get("notes"):
                print(f"      {produce['notes']}")

    # Get what's coming soon (next 2 months)
    coming_soon = []
    for produce in seasonal_produce:
        next_months = [(current_month % 12) + 1, (current_month % 12) + 2]
        if any(m in produce["available_months"] for m in next_months):
            if produce not in in_season_now:
                coming_soon.append(produce)

    if coming_soon:
        print(f"\n🔜 COMING SOON ({len(coming_soon)} items):")
        for produce in coming_soon:
            months = produce["available_months"]
            month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            month_str = ", ".join([month_names[m] for m in months])
            print(f"   • {produce['name']} ({month_str})")


def print_local_farms(stores):
    """Print local farm information"""
    farms = [s for s in stores if s.is_local_farm]
    if not farms:
        return

    print("\n" + "=" * 80)
    print("🚜 LOCAL FARMS & MARKETS")
    print("=" * 80)

    from datetime import date
    current_month = date.today().month

    for farm in sorted(farms, key=lambda f: f.priority):
        # Check if currently open
        is_open = not farm.seasonal_availability or current_month in farm.seasonal_availability
        status = "🟢 OPEN NOW" if is_open else "🔴 CLOSED (seasonal)"

        print(f"\n{status} - {farm.name}")
        print(f"   Priority: {farm.priority}")

        if farm.specialties:
            print(f"   Specialties: {', '.join(farm.specialties)}")

        if farm.seasonal_availability:
            month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            months = [month_names[m] for m in farm.seasonal_availability]
            print(f"   Open: {', '.join(months)}")

        if farm.notes:
            print(f"   ℹ️  {farm.notes}")


def plan_thanksgiving():
    """Generate complete Thanksgiving plan"""
    print("🦃" * 40)
    print("SOUS - Thanksgiving 2025 Kitchen Manager")
    print("🦃" * 40)

    # Load data
    print("\n📥 Loading data...")
    event, recipes, stores, ingredients, inventory_items, leftovers, waste_log, garden_harvests, seasonal_produce = load_data()
    print(f"   ✓ Loaded {len(recipes)} recipes")
    print(f"   ✓ Loaded {len(stores)} stores ({sum(1 for s in stores if s.is_local_farm)} local farms)")
    print(f"   ✓ Loaded {len(ingredients)} ingredients")
    print(f"   ✓ Loaded {len(inventory_items)} inventory items")
    print(f"   ✓ Loaded {len(leftovers)} leftovers")
    print(f"   ✓ Loaded {len(waste_log)} waste log entries")
    print(f"   ✓ Loaded {len(garden_harvests)} garden harvests")
    print(f"   ✓ Loaded {len(seasonal_produce)} seasonal produce items")
    print(f"   ✓ Event: {event.name} on {event.date.strftime('%A, %B %d, %Y')}")
    print(f"   ✓ Prep days: {len(event.prep_days)} days")

    # Create inventory manager with waste tracking
    inventory_manager = InventoryManager(inventory_items, waste_log)

    # Show current inventory
    print_inventory(inventory_manager, ingredients)

    # Generate shopping lists (accounting for inventory)
    print("\n🛒 Generating shopping lists (accounting for inventory)...")
    shopping_gen = ShoppingListGenerator()
    shopping_lists = shopping_gen.generate_shopping_lists(
        recipes, ingredients, stores, inventory_manager=inventory_manager
    )
    print(f"   ✓ Shopping lists generated for {len(shopping_lists)} stores")

    # Calculate items saved by having inventory
    shopping_lists_no_inventory = shopping_gen.generate_shopping_lists(
        recipes, ingredients, stores
    )
    items_saved = sum(len(items) for items in shopping_lists_no_inventory.values()) - sum(len(items) for items in shopping_lists.values())
    if items_saved > 0:
        print(f"   ✓ {items_saved} items skipped (already in inventory!)")

    # Print shopping lists
    print_shopping_lists(shopping_lists, stores)

    # Estimate shopping time
    time_estimates = shopping_gen.estimate_shopping_time(shopping_lists)
    total_shopping_time = sum(time_estimates.values())
    print(f"\n⏱️  Estimated total shopping time: {total_shopping_time} minutes ({total_shopping_time // 60}h {total_shopping_time % 60}m)")

    # Generate schedule
    print("\n📆 Generating prep schedule...")
    scheduler = Scheduler()
    daily_plans = scheduler.generate_daily_plans(event, recipes)
    print(f"   ✓ Schedule generated for {len(daily_plans)} days")

    # Print schedule
    print_daily_plans(daily_plans, recipes)

    # Check for conflicts
    print("\n🔍 Checking for appliance conflicts...")
    conflicts = scheduler.detect_conflicts(daily_plans, event)
    print_conflicts(conflicts)

    # Show leftovers (if any)
    print_leftovers(leftovers)

    # Show waste statistics and learnings
    print_waste_statistics(inventory_manager, ingredients)

    # Show garden harvests
    print_garden_harvests(garden_harvests, ingredients)

    # Show seasonal produce calendar
    print_seasonal_calendar(seasonal_produce)

    # Show local farms
    print_local_farms(stores)

    # Summary
    total_work = sum(plan.total_duration_minutes() for plan in daily_plans)
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"Total prep work: {total_work} minutes ({total_work // 60}h {total_work % 60}m)")
    print(f"Shopping time: {total_shopping_time} minutes")
    print(f"Total stores: {len(shopping_lists)}")
    print(f"Total recipes: {len(recipes)}")
    print(f"Optional recipes: {sum(1 for r in recipes if r.is_optional)}")
    print(f"Inventory items: {len(inventory_items)}")
    if items_saved > 0:
        print(f"Items saved (in stock): {items_saved}")
    print("\n✅ Thanksgiving plan complete!")
    print("=" * 80 + "\n")


def main():
    """Main CLI entry point"""
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help", "help"]:
        print("SOUS - Kitchen Manager CLI")
        print("\nUsage:")
        print("  python app.py              Run Thanksgiving planner")
        print("  python -m sous.app         Run from package")
        print("\nOptions:")
        print("  -h, --help                 Show this help message")
        print("  --ai                       Enable HoloLoom AI integration")
        print("  --voice                    Enable voice assistant")
        print("  --demo                     Run HoloLoom integration demo")
        return

    # Check for AI integration flags
    enable_ai = "--ai" in sys.argv
    enable_voice = "--voice" in sys.argv
    run_demo = "--demo" in sys.argv

    if run_demo:
        # Run AI integration demo
        import asyncio
        print("\n🤖 Running HoloLoom integration demo...")
        print("   (This requires HoloLoom to be installed)\n")

        try:
            from sous.services.hololoom_bridge import HOLOLOOM_AVAILABLE
            if not HOLOLOOM_AVAILABLE:
                print("⚠️  HoloLoom not available. Install with:")
                print("   pip install -e HoloLoom\n")
                return

            # Run quick demo
            asyncio.run(demo_ai_integration())
        except Exception as e:
            print(f"❌ Error running demo: {e}")
        return

    if enable_ai or enable_voice:
        print("\n🤖 AI features enabled!")
        print("   Memory: Stores cooking experiences")
        print("   RAG: Intelligent recipe suggestions")
        print("   Agentic: Smart meal planning")
        if enable_voice:
            print("   Voice: Hands-free cooking assistance")
        print()

    plan_thanksgiving()


async def demo_ai_integration():
    """Quick demo of AI integration"""
    from sous.services.hololoom_bridge import SousHoloLoomBridge

    async with SousHoloLoomBridge(enable_rag=True) as bridge:
        print("✓ HoloLoom bridge initialized\n")

        # Demo: Store recipe memory
        print("📖 Storing recipe in AI memory...")
        result = await bridge.remember_recipe({
            "name": "Thanksgiving Turkey",
            "description": "Classic roasted turkey",
            "prep_time": 30,
            "ingredients": ["turkey", "butter", "herbs", "salt", "pepper"]
        })
        print(f"   ✓ Memory ID: {result.get('memory_id', 'N/A')}\n")

        # Demo: Ask cooking question
        print("❓ Asking AI for recipe suggestion...")
        result = await bridge.suggest_recipes_for_ingredients(
            ingredients=["turkey", "potatoes", "green beans"],
            dietary_restrictions=None
        )
        if result['status'] == 'success':
            print(f"   AI Response: {result['response'][:150]}...")
            print(f"   Confidence: {result['confidence']:.2%}\n")

        print("✅ AI integration demo complete!\n")
        print("   Run full demo with:")
        print("   python demo_hololoom_integration.py")


if __name__ == "__main__":
    main()
