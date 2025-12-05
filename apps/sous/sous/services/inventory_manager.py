"""Inventory management service"""

from typing import List, Dict, Optional
from datetime import date, timedelta
from sous.models.inventory import (
    InventoryItem,
    Leftover,
    WastedItem,
    InventoryLocation,
    InventoryUnit,
    WasteReason,
)
from sous.models.recipe import Recipe


class InventoryManager:
    """
    Manages pantry, fridge, and leftover inventory

    Responsibilities:
    - Track what's in stock
    - Check expiration dates
    - Subtract from shopping lists
    - Log leftovers after meals
    - Track food waste and learn shelf life patterns
    - Monitor costs and waste statistics
    """

    def __init__(self, inventory: Optional[List[InventoryItem]] = None, waste_log: Optional[List[WastedItem]] = None):
        """
        Initialize inventory manager

        Args:
            inventory: Initial inventory items (default: empty)
            waste_log: Initial waste log (default: empty)
        """
        self.inventory: List[InventoryItem] = inventory or []
        self.waste_log: List[WastedItem] = waste_log or []

    def add_item(self, item: InventoryItem) -> None:
        """Add an item to inventory"""
        self.inventory.append(item)

    def remove_item(self, ingredient_id: str, quantity: float) -> bool:
        """
        Remove quantity from inventory

        Args:
            ingredient_id: Ingredient to remove
            quantity: Amount to remove

        Returns:
            True if successful, False if not enough in stock
        """
        # Find matching items
        matching_items = [
            item for item in self.inventory if item.ingredient_id == ingredient_id
        ]

        if not matching_items:
            return False

        # Sort by expiration (use oldest first)
        matching_items.sort(
            key=lambda x: x.expiration_date if x.expiration_date else date.max
        )

        # Subtract from items until quantity is met
        remaining = quantity
        for item in matching_items:
            if remaining <= 0:
                break

            if item.quantity >= remaining:
                item.quantity -= remaining
                remaining = 0
                # Remove if depleted
                if item.quantity == 0:
                    self.inventory.remove(item)
            else:
                remaining -= item.quantity
                self.inventory.remove(item)

        return remaining == 0

    def get_quantity(self, ingredient_id: str) -> float:
        """Get total quantity of an ingredient in inventory"""
        total = 0.0
        for item in self.inventory:
            if item.ingredient_id == ingredient_id:
                total += item.quantity
        return total

    def has_ingredient(self, ingredient_id: str, quantity: float = 1.0) -> bool:
        """Check if we have enough of an ingredient"""
        return self.get_quantity(ingredient_id) >= quantity

    def get_items_by_location(
        self, location: InventoryLocation
    ) -> List[InventoryItem]:
        """Get all items in a specific location"""
        return [item for item in self.inventory if item.location == location]

    def get_expiring_soon(self, days: int = 3) -> List[InventoryItem]:
        """
        Get items expiring within N days

        Args:
            days: Number of days to look ahead (default: 3)

        Returns:
            List of items expiring soon
        """
        cutoff = date.today() + timedelta(days=days)
        expiring = []

        for item in self.inventory:
            if item.expiration_date and item.expiration_date <= cutoff:
                expiring.append(item)

        # Sort by expiration date (soonest first)
        expiring.sort(key=lambda x: x.expiration_date or date.max)
        return expiring

    def get_expired(self) -> List[InventoryItem]:
        """Get all expired items"""
        return [item for item in self.inventory if item.is_expired()]

    def clean_expired(self) -> int:
        """
        Remove all expired items from inventory

        Returns:
            Number of items removed
        """
        expired = self.get_expired()
        for item in expired:
            self.inventory.remove(item)
        return len(expired)

    def adjust_shopping_list(
        self, shopping_lists: Dict[str, List[str]], ingredients: dict
    ) -> Dict[str, List[str]]:
        """
        Subtract inventory from shopping lists

        Args:
            shopping_lists: Original shopping lists (store_id -> [ingredient_names])
            ingredients: Ingredient database (ingredient_id -> details)

        Returns:
            Adjusted shopping lists (items we actually need to buy)
        """
        adjusted = {}

        for store_id, ingredient_names in shopping_lists.items():
            needed = []

            for ingredient_name in ingredient_names:
                # Find ingredient_id from name
                ingredient_id = None
                for ing_id, ing_data in ingredients.items():
                    if ing_data["name"] == ingredient_name:
                        ingredient_id = ing_id
                        break

                if not ingredient_id:
                    # Can't find ingredient, keep it in list
                    needed.append(ingredient_name)
                    continue

                # Check if we have it in inventory
                if not self.has_ingredient(ingredient_id):
                    needed.append(ingredient_name)

            if needed:
                adjusted[store_id] = needed

        return adjusted

    def log_leftover(
        self,
        recipe: Recipe,
        quantity: float,
        unit: InventoryUnit,
        shelf_life_days: int = 3,
    ) -> Leftover:
        """
        Log a leftover after cooking

        Args:
            recipe: Recipe that was cooked
            quantity: Amount of leftovers
            unit: Measurement unit
            shelf_life_days: How long it's good for (default: 3 days)

        Returns:
            Created leftover object
        """
        leftover = Leftover(
            recipe_id=recipe.id,
            description=recipe.name,
            quantity=quantity,
            unit=unit,
            created_date=date.today(),
            expiration_date=date.today() + timedelta(days=shelf_life_days),
            location=InventoryLocation.FRIDGE,
        )
        return leftover

    def summary(self) -> str:
        """Generate human-readable inventory summary"""
        lines = []
        lines.append(f"Total items in inventory: {len(self.inventory)}")

        # By location
        for location in InventoryLocation:
            items = self.get_items_by_location(location)
            if items:
                lines.append(f"  {location.value.capitalize()}: {len(items)} items")

        # Expiring soon
        expiring = self.get_expiring_soon(days=3)
        if expiring:
            lines.append(f"\n⚠️  {len(expiring)} items expiring within 3 days:")
            for item in expiring[:5]:  # Show first 5
                days_left = item.days_until_expiration()
                lines.append(f"  - {item.ingredient_id} ({days_left} days)")

        # Expired
        expired = self.get_expired()
        if expired:
            lines.append(f"\n❌ {len(expired)} expired items (should be removed)")

        return "\n".join(lines)

    # === WASTE TRACKING METHODS ===

    def log_waste(
        self,
        ingredient_id: str,
        description: str,
        quantity_wasted: float,
        unit: InventoryUnit,
        reason: WasteReason,
        purchased_date: Optional[date] = None,
        expected_expiration: Optional[date] = None,
        cost: Optional[float] = None,
        notes: str = ""
    ) -> WastedItem:
        """
        Log when an item is wasted/thrown away

        This helps track:
        - Actual shelf life patterns
        - Cost of waste
        - Common reasons for waste

        Args:
            ingredient_id: ID of wasted ingredient
            description: What was wasted
            quantity_wasted: Amount thrown away
            unit: Measurement unit
            reason: Why it was wasted (WasteReason enum)
            purchased_date: When item was bought (for shelf life calculation)
            expected_expiration: What the expiration date was
            cost: Cost of the item
            notes: Additional notes

        Returns:
            Created WastedItem
        """
        wasted = WastedItem(
            ingredient_id=ingredient_id,
            description=description,
            quantity_wasted=quantity_wasted,
            unit=unit,
            wasted_date=date.today(),
            reason=reason,
            purchased_date=purchased_date,
            cost=cost,
            expected_expiration=expected_expiration,
            notes=notes,
        )

        self.waste_log.append(wasted)
        return wasted

    def get_waste_statistics(self, days: Optional[int] = None) -> Dict:
        """
        Calculate waste statistics

        Args:
            days: Only consider waste from last N days (None = all time)

        Returns:
            Dictionary with waste stats:
            - total_items_wasted: Count of items
            - total_cost: Total $ wasted
            - by_reason: Breakdown by waste reason
            - by_ingredient: Most commonly wasted ingredients
            - avg_shelf_life_days: Average actual shelf life
        """
        # Filter by date if specified
        waste_items = self.waste_log
        if days:
            cutoff = date.today() - timedelta(days=days)
            waste_items = [w for w in waste_items if w.wasted_date >= cutoff]

        if not waste_items:
            return {
                "total_items_wasted": 0,
                "total_cost": 0.0,
                "by_reason": {},
                "by_ingredient": {},
                "avg_shelf_life_days": None,
            }

        # Calculate stats
        total_cost = sum(w.cost for w in waste_items if w.cost)
        by_reason = {}
        by_ingredient = {}
        shelf_lives = []

        for item in waste_items:
            # By reason
            reason = item.reason.value
            if reason not in by_reason:
                by_reason[reason] = {"count": 0, "cost": 0.0}
            by_reason[reason]["count"] += 1
            by_reason[reason]["cost"] += item.cost or 0.0

            # By ingredient
            ing_id = item.ingredient_id
            if ing_id not in by_ingredient:
                by_ingredient[ing_id] = {"count": 0, "cost": 0.0}
            by_ingredient[ing_id]["count"] += 1
            by_ingredient[ing_id]["cost"] += item.cost or 0.0

            # Shelf life
            if item.actual_shelf_life_days():
                shelf_lives.append(item.actual_shelf_life_days())

        avg_shelf_life = sum(shelf_lives) / len(shelf_lives) if shelf_lives else None

        return {
            "total_items_wasted": len(waste_items),
            "total_cost": round(total_cost, 2),
            "by_reason": by_reason,
            "by_ingredient": by_ingredient,
            "avg_shelf_life_days": round(avg_shelf_life, 1) if avg_shelf_life else None,
        }

    def calculate_actual_shelf_life(self, ingredient_id: str) -> Optional[float]:
        """
        Calculate average actual shelf life for an ingredient based on waste history

        This learns how long items REALLY last in your household,
        which may differ from printed expiration dates.

        Args:
            ingredient_id: Ingredient to analyze

        Returns:
            Average shelf life in days (or None if no data)
        """
        matching = [
            w for w in self.waste_log
            if w.ingredient_id == ingredient_id and w.actual_shelf_life_days()
        ]

        if not matching:
            return None

        shelf_lives = [w.actual_shelf_life_days() for w in matching]
        return round(sum(shelf_lives) / len(shelf_lives), 1)

    def get_total_waste_cost(self, days: Optional[int] = None) -> float:
        """
        Calculate total cost of wasted food

        Args:
            days: Only consider waste from last N days (None = all time)

        Returns:
            Total dollar amount wasted
        """
        stats = self.get_waste_statistics(days=days)
        return stats["total_cost"]

    def get_most_wasted_ingredients(self, limit: int = 5) -> List[tuple]:
        """
        Get ingredients that are wasted most frequently

        Args:
            limit: Maximum number to return

        Returns:
            List of (ingredient_id, count, total_cost) tuples, sorted by count
        """
        stats = self.get_waste_statistics()
        by_ingredient = stats["by_ingredient"]

        # Sort by count
        sorted_items = sorted(
            by_ingredient.items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )

        return [
            (ing_id, data["count"], data["cost"])
            for ing_id, data in sorted_items[:limit]
        ]
