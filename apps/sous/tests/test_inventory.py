"""
Tests for Inventory Management System
"""

import sys
from pathlib import Path
from datetime import date, timedelta

# Add sous package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sous.models.inventory import (
    InventoryItem,
    Leftover,
    InventoryLocation,
    InventoryUnit,
)
from sous.services.inventory_manager import InventoryManager


def test_inventory_item_creation():
    """Test creating inventory items"""
    print("Testing inventory item creation...")

    item = InventoryItem(
        ingredient_id="onions",
        quantity=3.0,
        unit=InventoryUnit.ITEM,
        location=InventoryLocation.COUNTER,
    )

    assert item.ingredient_id == "onions"
    assert item.quantity == 3.0
    assert item.unit == InventoryUnit.ITEM
    assert item.location == InventoryLocation.COUNTER

    print("✓ Inventory item creation test passed")


def test_inventory_expiration():
    """Test expiration date logic"""
    print("Testing expiration dates...")

    # Item expiring tomorrow
    tomorrow = date.today() + timedelta(days=1)
    item = InventoryItem(
        ingredient_id="milk",
        quantity=1.0,
        unit=InventoryUnit.PACKAGE,
        location=InventoryLocation.FRIDGE,
        expiration_date=tomorrow,
    )

    assert not item.is_expired()
    assert item.days_until_expiration() == 1

    # Item expired yesterday
    yesterday = date.today() - timedelta(days=1)
    expired_item = InventoryItem(
        ingredient_id="old_cheese",
        quantity=1.0,
        unit=InventoryUnit.PACKAGE,
        location=InventoryLocation.FRIDGE,
        expiration_date=yesterday,
    )

    assert expired_item.is_expired()
    assert expired_item.days_until_expiration() == -1

    print("✓ Expiration date test passed")


def test_inventory_manager_add_remove():
    """Test adding and removing items from inventory"""
    print("Testing inventory manager add/remove...")

    manager = InventoryManager()

    # Add items
    item1 = InventoryItem(
        ingredient_id="onions",
        quantity=3.0,
        unit=InventoryUnit.ITEM,
        location=InventoryLocation.COUNTER,
    )
    manager.add_item(item1)

    assert manager.get_quantity("onions") == 3.0
    assert manager.has_ingredient("onions", 2.0)
    assert not manager.has_ingredient("onions", 5.0)

    # Remove some
    success = manager.remove_item("onions", 1.0)
    assert success
    assert manager.get_quantity("onions") == 2.0

    # Try to remove more than available
    success = manager.remove_item("onions", 5.0)
    assert not success  # Should fail
    assert manager.get_quantity("onions") == 2.0  # Quantity unchanged

    print("✓ Inventory manager add/remove test passed")


def test_inventory_by_location():
    """Test filtering items by location"""
    print("Testing inventory by location...")

    manager = InventoryManager()

    # Add items in different locations
    manager.add_item(
        InventoryItem(
            "onions",
            3.0,
            InventoryUnit.ITEM,
            InventoryLocation.COUNTER,
        )
    )
    manager.add_item(
        InventoryItem(
            "milk",
            1.0,
            InventoryUnit.PACKAGE,
            InventoryLocation.FRIDGE,
        )
    )
    manager.add_item(
        InventoryItem(
            "flour",
            2.0,
            InventoryUnit.LB,
            InventoryLocation.PANTRY,
        )
    )

    pantry_items = manager.get_items_by_location(InventoryLocation.PANTRY)
    assert len(pantry_items) == 1
    assert pantry_items[0].ingredient_id == "flour"

    fridge_items = manager.get_items_by_location(InventoryLocation.FRIDGE)
    assert len(fridge_items) == 1
    assert fridge_items[0].ingredient_id == "milk"

    print("✓ Inventory by location test passed")


def test_expiring_soon():
    """Test finding items expiring soon"""
    print("Testing expiring soon detection...")

    manager = InventoryManager()

    # Item expiring in 2 days
    soon = date.today() + timedelta(days=2)
    manager.add_item(
        InventoryItem(
            "milk",
            1.0,
            InventoryUnit.PACKAGE,
            InventoryLocation.FRIDGE,
            expiration_date=soon,
        )
    )

    # Item expiring in 10 days
    later = date.today() + timedelta(days=10)
    manager.add_item(
        InventoryItem(
            "cheese",
            1.0,
            InventoryUnit.PACKAGE,
            InventoryLocation.FRIDGE,
            expiration_date=later,
        )
    )

    expiring = manager.get_expiring_soon(days=3)
    assert len(expiring) == 1
    assert expiring[0].ingredient_id == "milk"

    print("✓ Expiring soon detection test passed")


def test_shopping_list_subtraction():
    """Test subtracting inventory from shopping lists"""
    print("Testing shopping list subtraction...")

    # Create inventory with some items
    manager = InventoryManager()
    manager.add_item(
        InventoryItem(
            "onions",
            3.0,
            InventoryUnit.ITEM,
            InventoryLocation.COUNTER,
        )
    )
    manager.add_item(
        InventoryItem(
            "salt",
            1.0,
            InventoryUnit.PACKAGE,
            InventoryLocation.PANTRY,
        )
    )

    # Original shopping list
    shopping_lists = {
        "store_amish": ["Onions", "Salt", "Cornbread"],
        "store_vuck": ["Parsley", "Sage"],
    }

    # Ingredient database
    ingredients = {
        "onions": {"name": "Onions"},
        "salt": {"name": "Salt"},
        "cornbread": {"name": "Cornbread"},
        "parsley": {"name": "Parsley"},
        "sage": {"name": "Sage"},
    }

    # Adjust shopping list
    adjusted = manager.adjust_shopping_list(shopping_lists, ingredients)

    # Should remove onions and salt (we have them)
    assert "Onions" not in adjusted.get("store_amish", [])
    assert "Salt" not in adjusted.get("store_amish", [])
    assert "Cornbread" in adjusted.get("store_amish", [])

    # Vuck items unchanged (we don't have them)
    assert "Parsley" in adjusted.get("store_vuck", [])
    assert "Sage" in adjusted.get("store_vuck", [])

    print("✓ Shopping list subtraction test passed")


def test_leftover_creation():
    """Test creating leftovers"""
    print("Testing leftover creation...")

    leftover = Leftover(
        recipe_id="recipe_stuffing",
        description="Herb Stuffing",
        quantity=4.0,
        unit=InventoryUnit.CUP,
        created_date=date.today(),
        expiration_date=date.today() + timedelta(days=3),
    )

    assert leftover.recipe_id == "recipe_stuffing"
    assert leftover.quantity == 4.0
    assert not leftover.consumed
    assert leftover.days_remaining() == 3

    print("✓ Leftover creation test passed")


def test_leftover_expiration():
    """Test leftover expiration logic"""
    print("Testing leftover expiration...")

    # Fresh leftover (3 days)
    fresh = Leftover(
        "recipe_mashed_potatoes",
        "Mashed Potatoes",
        2.0,
        InventoryUnit.CUP,
        date.today(),
        date.today() + timedelta(days=3),
    )
    assert not fresh.is_expired()

    # Expired leftover
    expired = Leftover(
        "recipe_old_food",
        "Old Food",
        1.0,
        InventoryUnit.CUP,
        date.today() - timedelta(days=5),
        date.today() - timedelta(days=2),
    )
    assert expired.is_expired()
    assert expired.days_remaining() < 0

    print("✓ Leftover expiration test passed")


def run_all_tests():
    """Run all inventory tests"""
    print("=" * 80)
    print("Inventory Management Tests")
    print("=" * 80 + "\n")

    tests = [
        test_inventory_item_creation,
        test_inventory_expiration,
        test_inventory_manager_add_remove,
        test_inventory_by_location,
        test_expiring_soon,
        test_shopping_list_subtraction,
        test_leftover_creation,
        test_leftover_expiration,
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
