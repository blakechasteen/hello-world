#!/usr/bin/env python3
"""
Community Inventory Manager - Multi-organization food inventory tracking

Manages food inventory across multiple organizations in the food sharing network:
- Track current stock levels
- Identify surplus and expiring items
- Generate donation listings automatically
- Provide inventory analytics
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from decimal import Decimal
from collections import defaultdict

from sous.models.organization import Organization
from sous.models.food_item import (
    FoodItem,
    FoodCategory,
    FoodSafetyRating,
    DonationListing,
    SurplusAlert,
    create_prepared_meal,
    create_produce_item
)


@dataclass
class InventoryStats:
    """Inventory statistics for an organization"""
    total_items: int = 0
    total_weight_lbs: Decimal = Decimal("0.0")
    total_value: Decimal = Decimal("0.0")

    items_expiring_today: int = 0
    items_expiring_this_week: int = 0

    by_category: Dict[FoodCategory, int] = field(default_factory=lambda: defaultdict(int))
    by_safety_rating: Dict[FoodSafetyRating, int] = field(default_factory=lambda: defaultdict(int))


class CommunityInventoryManager:
    """
    Manages food inventory across organizations in the sharing network.

    Tracks stock, identifies surplus, and coordinates donations.
    """

    def __init__(self):
        # Organization inventories: org_id -> {item_id -> FoodItem}
        self.inventories: Dict[str, Dict[str, FoodItem]] = defaultdict(dict)

        # Auto-donation settings: org_id -> settings
        self.auto_donation_settings: Dict[str, Dict] = {}

    # ========================================================================
    # Inventory Management
    # ========================================================================

    def add_item(
        self,
        org_id: str,
        item: FoodItem
    ):
        """Add item to organization's inventory"""
        if org_id not in self.inventories:
            self.inventories[org_id] = {}

        self.inventories[org_id][item.item_id] = item

    def remove_item(
        self,
        org_id: str,
        item_id: str
    ) -> Optional[FoodItem]:
        """Remove item from inventory (when donated or consumed)"""
        if org_id not in self.inventories:
            return None

        return self.inventories[org_id].pop(item_id, None)

    def get_item(
        self,
        org_id: str,
        item_id: str
    ) -> Optional[FoodItem]:
        """Get specific item from inventory"""
        return self.inventories.get(org_id, {}).get(item_id)

    def get_inventory(
        self,
        org_id: str
    ) -> List[FoodItem]:
        """Get all items in organization's inventory"""
        return list(self.inventories.get(org_id, {}).values())

    def update_item_quantity(
        self,
        org_id: str,
        item_id: str,
        new_quantity: Decimal
    ) -> bool:
        """Update quantity of item in inventory"""
        item = self.get_item(org_id, item_id)
        if not item:
            return False

        item.quantity = new_quantity

        # Remove if quantity is zero
        if new_quantity <= 0:
            self.remove_item(org_id, item_id)

        return True

    # ========================================================================
    # Surplus Identification
    # ========================================================================

    def identify_surplus(
        self,
        org_id: str,
        days_threshold: int = 2
    ) -> List[FoodItem]:
        """
        Identify surplus items that should be donated.

        Items are considered surplus if:
        - Expiring within days_threshold
        - Safety rating is FAIR or DONATE_IMMEDIATELY
        - Organization has marked them as surplus
        """
        inventory = self.get_inventory(org_id)
        surplus = []

        for item in inventory:
            # Check expiration
            days_until_exp = item.get_days_until_expiration()
            if days_until_exp is not None and days_until_exp <= days_threshold:
                surplus.append(item)
                continue

            # Check safety rating
            rating = item.get_safety_rating()
            if rating in [FoodSafetyRating.FAIR, FoodSafetyRating.DONATE_IMMEDIATELY]:
                surplus.append(item)

        return surplus

    def identify_expiring_items(
        self,
        org_id: str,
        days_threshold: int = 7
    ) -> List[FoodItem]:
        """Find items expiring within threshold"""
        inventory = self.get_inventory(org_id)
        expiring = []

        for item in inventory:
            days_until_exp = item.get_days_until_expiration()
            if days_until_exp is not None and 0 <= days_until_exp <= days_threshold:
                expiring.append(item)

        # Sort by days until expiration (soonest first)
        expiring.sort(key=lambda item: item.get_days_until_expiration() or 999)

        return expiring

    def get_unsafe_items(
        self,
        org_id: str
    ) -> List[FoodItem]:
        """Find items that are no longer safe to donate"""
        inventory = self.get_inventory(org_id)
        unsafe = []

        for item in inventory:
            if item.get_safety_rating() == FoodSafetyRating.NOT_SAFE:
                unsafe.append(item)

        return unsafe

    # ========================================================================
    # Automatic Donation
    # ========================================================================

    def configure_auto_donation(
        self,
        org_id: str,
        enabled: bool = True,
        days_before_expiration: int = 2,
        min_quantity_lbs: Decimal = Decimal("5.0"),
        categories: Optional[List[FoodCategory]] = None
    ):
        """
        Configure automatic donation listing creation.

        Args:
            org_id: Organization ID
            enabled: Enable auto-donation
            days_before_expiration: Days threshold for auto-donation
            min_quantity_lbs: Minimum quantity to trigger auto-donation
            categories: Food categories to include (None = all)
        """
        self.auto_donation_settings[org_id] = {
            "enabled": enabled,
            "days_before_expiration": days_before_expiration,
            "min_quantity_lbs": min_quantity_lbs,
            "categories": categories
        }

    def check_auto_donation_triggers(
        self,
        org_id: str
    ) -> List[FoodItem]:
        """
        Check if any items should trigger automatic donation.

        Returns list of items that meet auto-donation criteria.
        """
        settings = self.auto_donation_settings.get(org_id)
        if not settings or not settings["enabled"]:
            return []

        days_threshold = settings["days_before_expiration"]
        min_quantity = settings["min_quantity_lbs"]
        categories = settings["categories"]

        # Find surplus items
        surplus = self.identify_surplus(org_id, days_threshold)

        # Filter by settings
        triggered = []
        for item in surplus:
            # Check quantity
            if item.unit == "lbs" and item.quantity < min_quantity:
                continue

            # Check category
            if categories and item.category not in categories:
                continue

            triggered.append(item)

        return triggered

    def create_auto_donation_listing(
        self,
        org_id: str,
        items: List[FoodItem],
        hours_available: int = 6
    ) -> Optional[DonationListing]:
        """
        Create automatic donation listing for items.

        Returns DonationListing if created, None if no items.
        """
        if not items:
            return None

        # Create listing
        listing_id = f"auto_{org_id}_{datetime.now().timestamp()}"

        # Calculate availability window
        now = datetime.now()
        available_until = now + timedelta(hours=hours_available)

        # Find soonest expiration to use as upper bound
        soonest_expiration = min(
            item.expiration_date
            for item in items
            if item.expiration_date
        )
        if soonest_expiration and soonest_expiration < available_until:
            available_until = soonest_expiration

        listing = DonationListing(
            listing_id=listing_id,
            donor_org_id=org_id,
            items=items,
            available_from=now,
            available_until=available_until,
            notes="Auto-generated listing for surplus items",
            pickup_address=None,  # Would be set from organization
            delivery_available=False  # Default to pickup only
        )

        return listing

    # ========================================================================
    # Analytics
    # ========================================================================

    def get_inventory_stats(
        self,
        org_id: str
    ) -> InventoryStats:
        """Get comprehensive inventory statistics"""
        inventory = self.get_inventory(org_id)

        stats = InventoryStats()
        stats.total_items = len(inventory)

        for item in inventory:
            # Totals
            if item.unit == "lbs":
                stats.total_weight_lbs += item.quantity
            stats.total_value += item.estimated_value

            # Categories
            stats.by_category[item.category] += 1

            # Safety ratings
            rating = item.get_safety_rating()
            stats.by_safety_rating[rating] += 1

            # Expiration tracking
            days_until_exp = item.get_days_until_expiration()
            if days_until_exp is not None:
                if days_until_exp == 0:
                    stats.items_expiring_today += 1
                elif days_until_exp <= 7:
                    stats.items_expiring_this_week += 1

        return stats

    def get_expiration_forecast(
        self,
        org_id: str,
        days_ahead: int = 7
    ) -> Dict[str, List[FoodItem]]:
        """
        Forecast items expiring over next N days.

        Returns dict: "day_YYYY-MM-DD" -> [items expiring that day]
        """
        inventory = self.get_inventory(org_id)
        forecast = defaultdict(list)

        for item in inventory:
            if not item.expiration_date:
                continue

            exp_date = item.expiration_date.date()
            today = datetime.now().date()

            # Only include items expiring within forecast window
            days_until = (exp_date - today).days
            if 0 <= days_until <= days_ahead:
                key = f"day_{exp_date.isoformat()}"
                forecast[key].append(item)

        return dict(forecast)

    def get_category_breakdown(
        self,
        org_id: str
    ) -> Dict[FoodCategory, Dict]:
        """
        Get detailed breakdown by category.

        Returns: category -> {count, weight_lbs, value}
        """
        inventory = self.get_inventory(org_id)
        breakdown = defaultdict(lambda: {
            "count": 0,
            "weight_lbs": Decimal("0.0"),
            "value": Decimal("0.0")
        })

        for item in inventory:
            cat = item.category
            breakdown[cat]["count"] += 1

            if item.unit == "lbs":
                breakdown[cat]["weight_lbs"] += item.quantity

            breakdown[cat]["value"] += item.estimated_value

        return dict(breakdown)

    def get_waste_risk_items(
        self,
        org_id: str,
        risk_threshold_days: int = 2
    ) -> List[Dict]:
        """
        Identify items at risk of waste.

        Returns items with risk scores and recommendations.
        """
        expiring = self.identify_expiring_items(org_id, risk_threshold_days)
        at_risk = []

        for item in expiring:
            days_until_exp = item.get_days_until_expiration()
            if days_until_exp is None:
                continue

            # Calculate risk score (0-1, higher = more at risk)
            if days_until_exp == 0:
                risk_score = 1.0
                recommendation = "Donate immediately or discard"
            elif days_until_exp == 1:
                risk_score = 0.8
                recommendation = "Donate today"
            else:
                risk_score = 0.5
                recommendation = "Plan to donate soon"

            at_risk.append({
                "item": item,
                "days_until_expiration": days_until_exp,
                "risk_score": risk_score,
                "recommendation": recommendation
            })

        # Sort by risk (highest first)
        at_risk.sort(key=lambda x: x["risk_score"], reverse=True)

        return at_risk

    # ========================================================================
    # Batch Operations
    # ========================================================================

    def bulk_add_items(
        self,
        org_id: str,
        items: List[FoodItem]
    ):
        """Add multiple items to inventory at once"""
        for item in items:
            self.add_item(org_id, item)

    def bulk_remove_items(
        self,
        org_id: str,
        item_ids: List[str]
    ) -> List[FoodItem]:
        """Remove multiple items from inventory"""
        removed = []

        for item_id in item_ids:
            item = self.remove_item(org_id, item_id)
            if item:
                removed.append(item)

        return removed

    def transfer_items(
        self,
        from_org_id: str,
        to_org_id: str,
        item_ids: List[str]
    ) -> bool:
        """
        Transfer items between organizations.

        Returns True if all transfers successful.
        """
        items = []

        # Remove from source
        for item_id in item_ids:
            item = self.remove_item(from_org_id, item_id)
            if not item:
                # Rollback: add items back
                for rolled_back in items:
                    self.add_item(from_org_id, rolled_back)
                return False

            items.append(item)

        # Add to destination
        for item in items:
            # Update source org
            item.source_org_id = from_org_id
            self.add_item(to_org_id, item)

        return True

    # ========================================================================
    # Reporting
    # ========================================================================

    def generate_inventory_report(
        self,
        org_id: str
    ) -> Dict:
        """Generate comprehensive inventory report"""
        stats = self.get_inventory_stats(org_id)
        category_breakdown = self.get_category_breakdown(org_id)
        expiration_forecast = self.get_expiration_forecast(org_id)
        waste_risk = self.get_waste_risk_items(org_id)

        return {
            "org_id": org_id,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_items": stats.total_items,
                "total_weight_lbs": float(stats.total_weight_lbs),
                "total_value": float(stats.total_value),
                "items_expiring_today": stats.items_expiring_today,
                "items_expiring_this_week": stats.items_expiring_this_week
            },
            "by_category": {
                cat.value: {
                    "count": data["count"],
                    "weight_lbs": float(data["weight_lbs"]),
                    "value": float(data["value"])
                }
                for cat, data in category_breakdown.items()
            },
            "by_safety_rating": {
                rating.value: count
                for rating, count in stats.by_safety_rating.items()
            },
            "expiration_forecast": {
                day: len(items)
                for day, items in expiration_forecast.items()
            },
            "waste_risk_count": len(waste_risk),
            "high_risk_items": [
                {
                    "item_id": risk["item"].item_id,
                    "name": risk["item"].name,
                    "days_until_expiration": risk["days_until_expiration"],
                    "risk_score": risk["risk_score"],
                    "recommendation": risk["recommendation"]
                }
                for risk in waste_risk[:10]  # Top 10 high-risk items
            ]
        }

    def get_multi_org_summary(
        self,
        org_ids: Optional[List[str]] = None
    ) -> Dict:
        """Get summary across multiple organizations"""
        if org_ids is None:
            org_ids = list(self.inventories.keys())

        total_items = 0
        total_weight = Decimal("0.0")
        total_value = Decimal("0.0")
        items_at_risk = 0

        org_summaries = []

        for org_id in org_ids:
            stats = self.get_inventory_stats(org_id)
            waste_risk = self.get_waste_risk_items(org_id)

            total_items += stats.total_items
            total_weight += stats.total_weight_lbs
            total_value += stats.total_value
            items_at_risk += len(waste_risk)

            org_summaries.append({
                "org_id": org_id,
                "items": stats.total_items,
                "weight_lbs": float(stats.total_weight_lbs),
                "value": float(stats.total_value),
                "items_at_risk": len(waste_risk)
            })

        return {
            "total_organizations": len(org_ids),
            "total_items": total_items,
            "total_weight_lbs": float(total_weight),
            "total_value": float(total_value),
            "items_at_risk": items_at_risk,
            "organizations": org_summaries
        }
