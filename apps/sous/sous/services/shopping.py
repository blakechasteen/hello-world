"""
Shopping List Generator Service

Multi-store routing for ingredient shopping
"""

from typing import List, Dict, Optional
from sous.models.recipe import Recipe
from sous.models.store import Store


class ShoppingListGenerator:
    """
    Intelligent multi-store shopping list generator

    Handles:
    - Ingredient aggregation across recipes
    - Store routing based on priority
    - Store role matching (anchor, specialty, etc.)
    """

    def generate_shopping_lists(
        self,
        recipes: List[Recipe],
        ingredients: Dict[str, dict],
        stores: List[Store],
        inventory_manager=None,  # Optional: InventoryManager to check stock
    ) -> Dict[str, List[str]]:
        """
        Generate shopping lists grouped by store

        Algorithm:
        1. Aggregate all ingredient IDs from recipes
        2. Check inventory (if provided) to skip items we already have
        3. Route to stores based on priority
        4. Group by store ID

        Args:
            recipes: List of recipes to shop for
            ingredients: Dict mapping ingredient_id -> ingredient metadata
            stores: List of available stores
            inventory_manager: Optional InventoryManager to check stock

        Returns:
            Dict mapping store_id -> list of ingredient names
        """
        shopping_lists = {store.id: [] for store in stores}

        # Aggregate ingredients
        all_ingredient_ids = set()
        for recipe in recipes:
            for ing in recipe.ingredients:
                if not ing.optional:  # Skip optional ingredients
                    all_ingredient_ids.add(ing.ingredient_id)

        # Route each ingredient to a store
        for ing_id in all_ingredient_ids:
            if ing_id not in ingredients:
                print(f"Warning: Ingredient '{ing_id}' not found in ingredients database")
                continue

            # Check if we already have it in inventory
            if inventory_manager and inventory_manager.has_ingredient(ing_id):
                continue  # Skip - we have it!

            ing_data = ingredients[ing_id]
            store_id = self._select_store(ing_data, stores)

            if store_id:
                ing_name = ing_data.get("name", ing_id)
                shopping_lists[store_id].append(ing_name)

        # Remove empty lists
        shopping_lists = {
            store_id: items
            for store_id, items in shopping_lists.items()
            if items
        }

        return shopping_lists

    def _select_store(
        self, ingredient: dict, stores: List[Store]
    ) -> str:
        """
        Select best store for ingredient

        Priority logic:
        1. If ingredient has preferred_store, use that
        2. Otherwise, choose from possible_stores by priority
        3. If no possible_stores, use highest priority store

        Args:
            ingredient: Ingredient metadata dict
            stores: List of available stores

        Returns:
            Store ID
        """
        # Check for preferred store
        if "preferred_store" in ingredient:
            return ingredient["preferred_store"]

        # Check possible stores
        possible_store_ids = ingredient.get("possible_stores", [])
        if possible_store_ids:
            # Filter to stores that exist
            possible_stores = [s for s in stores if s.id in possible_store_ids]

            if possible_stores:
                # Sort by priority and return best
                possible_stores.sort(key=lambda s: s.priority)
                return possible_stores[0].id

        # Fallback: Use highest priority store
        if stores:
            stores_sorted = sorted(stores, key=lambda s: s.priority)
            return stores_sorted[0].id

        return None

    def optimize_route(
        self,
        shopping_lists: Dict[str, List[str]],
        stores: List[Store],
    ) -> List[Store]:
        """
        Suggest optimal store visit order

        For MVP: Simple priority-based ordering
        Future: Geographic routing, time optimization

        Args:
            shopping_lists: Shopping lists by store ID
            stores: List of stores

        Returns:
            Ordered list of stores to visit
        """
        # Get stores that have items
        stores_to_visit = []
        for store in stores:
            if store.id in shopping_lists and shopping_lists[store.id]:
                stores_to_visit.append(store)

        # Sort by priority
        stores_to_visit.sort(key=lambda s: s.priority)

        return stores_to_visit

    def estimate_shopping_time(
        self,
        shopping_lists: Dict[str, List[str]],
        minutes_per_store_base: int = 30,
        minutes_per_item: int = 2,
    ) -> Dict[str, int]:
        """
        Estimate shopping time per store

        Args:
            shopping_lists: Shopping lists by store ID
            minutes_per_store_base: Base time per store (checkout, parking, etc.)
            minutes_per_item: Time per item

        Returns:
            Dict mapping store_id -> estimated minutes
        """
        estimates = {}

        for store_id, items in shopping_lists.items():
            base_time = minutes_per_store_base
            item_time = len(items) * minutes_per_item
            total_time = base_time + item_time
            estimates[store_id] = total_time

        return estimates

    def generate_consolidated_list(
        self,
        shopping_lists: Dict[str, List[str]],
        stores: List[Store],
    ) -> List[Dict]:
        """
        Generate consolidated shopping list with store metadata

        Returns:
            List of dicts with store info + items
        """
        consolidated = []

        # Get store objects by ID
        store_map = {s.id: s for s in stores}

        for store_id, items in shopping_lists.items():
            if not items:
                continue

            store = store_map.get(store_id)
            if not store:
                continue

            consolidated.append({
                "store_id": store_id,
                "store_name": store.name,
                "priority": store.priority,
                "roles": store.roles,
                "items": sorted(items),  # Alphabetical for easy shopping
                "item_count": len(items),
            })

        # Sort by priority
        consolidated.sort(key=lambda x: x["priority"])

        return consolidated
