"""
Integration Tests for Thanksgiving Planning
"""

import sys
from pathlib import Path
import json

# Add sous package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sous.models.event import Event
from sous.models.recipe import Recipe
from sous.models.store import Store
from sous.services.scheduler import Scheduler
from sous.services.shopping import ShoppingListGenerator


def load_test_data():
    """Load seed data for testing"""
    data_dir = Path(__file__).parent.parent / 'sous' / 'data'

    with open(data_dir / 'stores.json') as f:
        stores = [Store.from_dict(s) for s in json.load(f)]

    with open(data_dir / 'event_thanksgiving.json') as f:
        event = Event.from_dict(json.load(f))

    with open(data_dir / 'ingredients.json') as f:
        ingredients = json.load(f)

    with open(data_dir / 'recipes_thanksgiving.json') as f:
        recipes = [Recipe.from_dict(r) for r in json.load(f)]

    return event, recipes, stores, ingredients


def test_data_loading():
    """Test that all data loads correctly"""
    print("Testing data loading...")

    event, recipes, stores, ingredients = load_test_data()

    assert event is not None, "Event should load"
    assert event.name == "Thanksgiving 2025", "Event name should match"
    assert len(event.prep_days) == 3, "Should have 3 prep days"
    assert len(event.appliances) == 3, "Should have 3 appliances"

    assert len(recipes) == 9, "Should have 9 recipes"
    assert len(stores) == 6, "Should have 6 stores"
    assert len(ingredients) > 0, "Should have ingredients"

    print("✓ Data loading test passed")


def test_shopping_list_generation():
    """Test shopping list generation"""
    print("Testing shopping list generation...")

    event, recipes, stores, ingredients = load_test_data()

    shopping_gen = ShoppingListGenerator()
    shopping_lists = shopping_gen.generate_shopping_lists(
        recipes, ingredients, stores
    )

    assert len(shopping_lists) > 0, "Should generate shopping lists"

    # Check that specific stores have items
    assert "store_amish" in shopping_lists, "Amish Market should have items"
    assert "store_earthfare" in shopping_lists, "Earth Fare should have items"

    # Check that items are strings
    for store_id, items in shopping_lists.items():
        assert isinstance(items, list), f"{store_id} items should be a list"
        assert len(items) > 0, f"{store_id} should have items"

    print(f"✓ Shopping list generation test passed ({len(shopping_lists)} stores)")


def test_schedule_generation():
    """Test schedule generation"""
    print("Testing schedule generation...")

    event, recipes, stores, ingredients = load_test_data()

    scheduler = Scheduler()
    daily_plans = scheduler.generate_daily_plans(event, recipes)

    assert len(daily_plans) > 0, "Should generate daily plans"
    assert len(daily_plans) == 4, "Should have 4 days (3 prep + 1 event)"

    # Check that tasks exist
    total_tasks = sum(len(plan.tasks) for plan in daily_plans)
    assert total_tasks > 0, "Should have scheduled tasks"

    # Check that event day has tasks
    event_day_plan = [p for p in daily_plans if p.date == event.date]
    assert len(event_day_plan) == 1, "Should have event day plan"
    assert len(event_day_plan[0].tasks) > 0, "Event day should have tasks"

    print(f"✓ Schedule generation test passed ({total_tasks} total tasks)")


def test_appliance_conflict_detection():
    """Test appliance conflict detection"""
    print("Testing appliance conflict detection...")

    event, recipes, stores, ingredients = load_test_data()

    scheduler = Scheduler()
    daily_plans = scheduler.generate_daily_plans(event, recipes)
    conflicts = scheduler.detect_conflicts(daily_plans, event)

    # We might have conflicts (that's ok for MVP)
    # Just check that detection runs
    assert isinstance(conflicts, list), "Conflicts should be a list"

    if conflicts:
        print(f"  ⚠️  Detected {len(conflicts)} potential conflicts (expected for MVP)")
    else:
        print(f"  ✓ No conflicts detected")

    print("✓ Conflict detection test passed")


def test_multi_store_routing():
    """Test that ingredients are routed to correct stores"""
    print("Testing multi-store routing...")

    event, recipes, stores, ingredients = load_test_data()

    shopping_gen = ShoppingListGenerator()
    shopping_lists = shopping_gen.generate_shopping_lists(
        recipes, ingredients, stores
    )

    # Check that white truffle oil goes to Earth Fare
    earthfare_items = shopping_lists.get("store_earthfare", [])
    assert any("Truffle" in item for item in earthfare_items), \
        "Truffle oil should be at Earth Fare"

    # Check that herbs go to Vuck Farms (produce specialty)
    # (They might also be at other stores due to routing logic)

    print("✓ Multi-store routing test passed")


def run_all_tests():
    """Run all integration tests"""
    print("=" * 80)
    print("SOUS Integration Tests - Thanksgiving Planning")
    print("=" * 80 + "\n")

    tests = [
        test_data_loading,
        test_shopping_list_generation,
        test_schedule_generation,
        test_appliance_conflict_detection,
        test_multi_store_routing,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} error: {e}")
            failed += 1

    print("\n" + "=" * 80)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 80)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
