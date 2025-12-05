#!/usr/bin/env python3
"""
Food Item Models for Community Food Sharing

Tracks individual food items, donations, and surplus alerts.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Dict
from decimal import Decimal


class FoodCategory(Enum):
    """Food categories for classification"""
    PRODUCE = "produce"
    DAIRY = "dairy"
    MEAT = "meat"
    BAKERY = "bakery"
    CANNED = "canned"
    FROZEN = "frozen"
    PREPARED_MEAL = "prepared_meal"
    PANTRY_STAPLE = "pantry_staple"
    BEVERAGE = "beverage"
    SNACK = "snack"


class StorageRequirement(Enum):
    """Storage temperature requirements"""
    ROOM_TEMP = "room_temp"
    REFRIGERATED = "refrigerated"
    FROZEN = "frozen"


class DonationStatus(Enum):
    """Status of donation listing"""
    AVAILABLE = "available"
    CLAIMED = "claimed"
    IN_TRANSIT = "in_transit"
    DELIVERED = "delivered"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class FoodSafetyRating(Enum):
    """Food safety rating"""
    EXCELLENT = "excellent"  # Just prepared/packaged
    GOOD = "good"  # Within optimal freshness
    FAIR = "fair"  # Near expiration but safe
    DONATE_IMMEDIATELY = "donate_immediately"  # Use today
    NOT_SAFE = "not_safe"  # Do not donate


@dataclass
class NutritionalInfo:
    """Nutritional information per serving"""
    serving_size: str = ""
    calories: Optional[int] = None
    protein_g: Optional[float] = None
    carbs_g: Optional[float] = None
    fat_g: Optional[float] = None
    fiber_g: Optional[float] = None
    sodium_mg: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "serving_size": self.serving_size,
            "calories": self.calories,
            "protein_g": self.protein_g,
            "carbs_g": self.carbs_g,
            "fat_g": self.fat_g,
            "fiber_g": self.fiber_g,
            "sodium_mg": self.sodium_mg
        }


@dataclass
class FoodItem:
    """
    Individual food item in the donation system.

    Represents a specific quantity of food with metadata.
    """
    item_id: str
    name: str
    category: FoodCategory
    description: str = ""

    # Quantity
    quantity: Decimal = Decimal("1.0")
    unit: str = "lbs"  # lbs, oz, count, servings, etc.
    servings: Optional[int] = None

    # Storage and safety
    storage_requirement: StorageRequirement = StorageRequirement.ROOM_TEMP
    expiration_date: Optional[datetime] = None
    prepared_date: Optional[datetime] = None
    use_by_date: Optional[datetime] = None

    # Nutrition
    nutrition: Optional[NutritionalInfo] = None

    # Origin
    source_org_id: str = ""
    donated_by: Optional[str] = None  # Person/department who donated

    # Dietary
    is_vegetarian: bool = False
    is_vegan: bool = False
    is_gluten_free: bool = False
    is_dairy_free: bool = False
    is_nut_free: bool = False
    allergens: List[str] = field(default_factory=list)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    estimated_value: Decimal = field(default_factory=lambda: Decimal("0.0"))

    def get_safety_rating(self) -> FoodSafetyRating:
        """Determine food safety rating based on dates"""
        now = datetime.now()

        # Check if already expired
        if self.expiration_date and now > self.expiration_date:
            return FoodSafetyRating.NOT_SAFE

        if self.use_by_date and now > self.use_by_date:
            return FoodSafetyRating.NOT_SAFE

        # Check freshness for prepared food
        if self.prepared_date:
            hours_since_prep = (now - self.prepared_date).total_seconds() / 3600

            if hours_since_prep < 2:
                return FoodSafetyRating.EXCELLENT
            elif hours_since_prep < 4:
                return FoodSafetyRating.GOOD
            elif hours_since_prep < 6:
                return FoodSafetyRating.FAIR
            else:
                return FoodSafetyRating.DONATE_IMMEDIATELY

        # Check days until expiration
        if self.expiration_date:
            days_until_expiration = (self.expiration_date - now).days

            if days_until_expiration > 7:
                return FoodSafetyRating.EXCELLENT
            elif days_until_expiration > 3:
                return FoodSafetyRating.GOOD
            elif days_until_expiration > 1:
                return FoodSafetyRating.FAIR
            else:
                return FoodSafetyRating.DONATE_IMMEDIATELY

        # No date information - assume good
        return FoodSafetyRating.GOOD

    def get_days_until_expiration(self) -> Optional[int]:
        """Get days until expiration (negative if expired)"""
        if not self.expiration_date:
            return None

        delta = self.expiration_date - datetime.now()
        return delta.days

    def is_safe_to_donate(self) -> bool:
        """Check if item is safe to donate"""
        rating = self.get_safety_rating()
        return rating != FoodSafetyRating.NOT_SAFE

    def matches_dietary_restrictions(self, restrictions: List[str]) -> bool:
        """Check if item matches dietary restrictions"""
        restriction_map = {
            "vegetarian": self.is_vegetarian,
            "vegan": self.is_vegan,
            "gluten_free": self.is_gluten_free,
            "dairy_free": self.is_dairy_free,
            "nut_free": self.is_nut_free
        }

        for restriction in restrictions:
            if restriction.lower() in restriction_map:
                if not restriction_map[restriction.lower()]:
                    return False

        return True

    def to_dict(self) -> dict:
        """Serialize to dictionary"""
        return {
            "item_id": self.item_id,
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "quantity": float(self.quantity),
            "unit": self.unit,
            "servings": self.servings,
            "storage_requirement": self.storage_requirement.value,
            "expiration_date": self.expiration_date.isoformat() if self.expiration_date else None,
            "prepared_date": self.prepared_date.isoformat() if self.prepared_date else None,
            "use_by_date": self.use_by_date.isoformat() if self.use_by_date else None,
            "nutrition": self.nutrition.to_dict() if self.nutrition else None,
            "dietary": {
                "is_vegetarian": self.is_vegetarian,
                "is_vegan": self.is_vegan,
                "is_gluten_free": self.is_gluten_free,
                "is_dairy_free": self.is_dairy_free,
                "is_nut_free": self.is_nut_free,
                "allergens": self.allergens
            },
            "safety_rating": self.get_safety_rating().value,
            "days_until_expiration": self.get_days_until_expiration(),
            "estimated_value": float(self.estimated_value),
            "created_at": self.created_at.isoformat()
        }


@dataclass
class DonationListing:
    """
    Public listing of food available for donation.

    Donors create listings, recipients browse and claim them.
    """
    listing_id: str
    donor_org_id: str
    items: List[FoodItem]

    # Availability
    available_from: datetime
    available_until: datetime
    status: DonationStatus = DonationStatus.AVAILABLE

    # Pickup/delivery
    pickup_address: Optional[str] = None
    delivery_available: bool = False
    delivery_radius_miles: float = 5.0

    # Recipient
    claimed_by_org_id: Optional[str] = None
    claimed_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    notes: str = ""
    special_instructions: str = ""

    # Analytics
    views_count: int = 0
    claims_count: int = 0  # Tracks number of claims (for tracking interest)

    def get_total_weight_lbs(self) -> Decimal:
        """Calculate total weight of all items"""
        total = Decimal("0.0")

        for item in self.items:
            if item.unit == "lbs":
                total += item.quantity
            elif item.unit == "oz":
                total += item.quantity / Decimal("16.0")
            # For other units, would need conversion logic

        return total

    def get_total_servings(self) -> int:
        """Calculate total servings across all items"""
        total = 0

        for item in self.items:
            if item.servings:
                total += item.servings

        return total

    def get_estimated_value(self) -> Decimal:
        """Calculate total estimated value"""
        return sum(item.estimated_value for item in self.items)

    def is_available(self) -> bool:
        """Check if listing is currently available"""
        if self.status != DonationStatus.AVAILABLE:
            return False

        now = datetime.now()
        return self.available_from <= now <= self.available_until

    def is_expired(self) -> bool:
        """Check if listing has expired"""
        return datetime.now() > self.available_until

    def has_dietary_match(self, restrictions: List[str]) -> bool:
        """Check if any items match dietary restrictions"""
        return any(item.matches_dietary_restrictions(restrictions) for item in self.items)

    def claim(self, recipient_org_id: str) -> bool:
        """
        Claim the listing for a recipient organization.

        Returns True if successfully claimed.
        """
        if not self.is_available():
            return False

        self.status = DonationStatus.CLAIMED
        self.claimed_by_org_id = recipient_org_id
        self.claimed_at = datetime.now()
        self.claims_count += 1

        return True

    def mark_in_transit(self) -> bool:
        """Mark listing as in transit"""
        if self.status != DonationStatus.CLAIMED:
            return False

        self.status = DonationStatus.IN_TRANSIT
        return True

    def mark_delivered(self) -> bool:
        """Mark listing as delivered"""
        if self.status != DonationStatus.IN_TRANSIT:
            return False

        self.status = DonationStatus.DELIVERED
        self.delivered_at = datetime.now()
        return True

    def cancel(self) -> bool:
        """Cancel the listing"""
        if self.status in [DonationStatus.DELIVERED, DonationStatus.EXPIRED]:
            return False

        self.status = DonationStatus.CANCELLED
        return True

    def to_dict(self) -> dict:
        """Serialize to dictionary"""
        return {
            "listing_id": self.listing_id,
            "donor_org_id": self.donor_org_id,
            "items": [item.to_dict() for item in self.items],
            "availability": {
                "available_from": self.available_from.isoformat(),
                "available_until": self.available_until.isoformat(),
                "status": self.status.value,
                "is_available": self.is_available(),
                "is_expired": self.is_expired()
            },
            "logistics": {
                "pickup_address": self.pickup_address,
                "delivery_available": self.delivery_available,
                "delivery_radius_miles": float(self.delivery_radius_miles)
            },
            "recipient": {
                "claimed_by_org_id": self.claimed_by_org_id,
                "claimed_at": self.claimed_at.isoformat() if self.claimed_at else None,
                "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None
            },
            "summary": {
                "total_weight_lbs": float(self.get_total_weight_lbs()),
                "total_servings": self.get_total_servings(),
                "estimated_value": float(self.get_estimated_value())
            },
            "metadata": {
                "created_at": self.created_at.isoformat(),
                "notes": self.notes,
                "special_instructions": self.special_instructions,
                "views_count": self.views_count,
                "claims_count": self.claims_count
            }
        }


@dataclass
class SurplusAlert:
    """
    Real-time alert for surplus food availability.

    Used when organizations have immediate surplus that needs
    quick pickup (e.g., restaurant end-of-day surplus).
    """
    alert_id: str
    donor_org_id: str
    items: List[FoodItem]

    # Urgency
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    urgent: bool = False  # Must be picked up ASAP

    # Notification
    notified_org_ids: List[str] = field(default_factory=list)
    responded_org_ids: List[str] = field(default_factory=list)

    # Resolution
    resolved: bool = False
    resolved_by_org_id: Optional[str] = None
    resolved_at: Optional[datetime] = None

    # Metadata
    notes: str = ""

    def is_active(self) -> bool:
        """Check if alert is still active"""
        if self.resolved:
            return False

        if self.expires_at and datetime.now() > self.expires_at:
            return False

        return True

    def get_minutes_until_expiration(self) -> Optional[int]:
        """Get minutes until alert expires"""
        if not self.expires_at:
            return None

        if self.resolved:
            return 0

        delta = self.expires_at - datetime.now()
        return max(0, int(delta.total_seconds() / 60))

    def respond(self, org_id: str):
        """Mark organization as responded to alert"""
        if org_id not in self.responded_org_ids:
            self.responded_org_ids.append(org_id)

    def resolve(self, org_id: str):
        """Resolve alert (someone claimed the surplus)"""
        self.resolved = True
        self.resolved_by_org_id = org_id
        self.resolved_at = datetime.now()

    def to_dict(self) -> dict:
        """Serialize to dictionary"""
        return {
            "alert_id": self.alert_id,
            "donor_org_id": self.donor_org_id,
            "items": [item.to_dict() for item in self.items],
            "urgency": {
                "created_at": self.created_at.isoformat(),
                "expires_at": self.expires_at.isoformat() if self.expires_at else None,
                "minutes_until_expiration": self.get_minutes_until_expiration(),
                "urgent": self.urgent,
                "is_active": self.is_active()
            },
            "notifications": {
                "notified_org_ids": self.notified_org_ids,
                "responded_org_ids": self.responded_org_ids,
                "response_rate": len(self.responded_org_ids) / max(1, len(self.notified_org_ids))
            },
            "resolution": {
                "resolved": self.resolved,
                "resolved_by_org_id": self.resolved_by_org_id,
                "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None
            },
            "notes": self.notes
        }


# Factory functions for creating food items

def create_prepared_meal(
    name: str,
    servings: int,
    description: str = "",
    prepared_date: Optional[datetime] = None,
    source_org_id: str = ""
) -> FoodItem:
    """Create a prepared meal item"""
    item_id = f"meal_{name.lower().replace(' ', '_')}_{datetime.now().timestamp()}"

    return FoodItem(
        item_id=item_id,
        name=name,
        category=FoodCategory.PREPARED_MEAL,
        description=description,
        quantity=Decimal(servings),
        unit="servings",
        servings=servings,
        storage_requirement=StorageRequirement.REFRIGERATED,
        prepared_date=prepared_date or datetime.now(),
        use_by_date=(prepared_date or datetime.now()) + timedelta(hours=6),
        source_org_id=source_org_id
    )


def create_produce_item(
    name: str,
    quantity_lbs: Decimal,
    days_until_expiration: int,
    source_org_id: str = ""
) -> FoodItem:
    """Create a produce item"""
    item_id = f"produce_{name.lower().replace(' ', '_')}_{datetime.now().timestamp()}"

    return FoodItem(
        item_id=item_id,
        name=name,
        category=FoodCategory.PRODUCE,
        quantity=quantity_lbs,
        unit="lbs",
        storage_requirement=StorageRequirement.REFRIGERATED,
        expiration_date=datetime.now() + timedelta(days=days_until_expiration),
        source_org_id=source_org_id,
        is_vegetarian=True,
        is_vegan=True
    )


def create_bakery_item(
    name: str,
    quantity: int,
    days_until_expiration: int,
    source_org_id: str = ""
) -> FoodItem:
    """Create a bakery item"""
    item_id = f"bakery_{name.lower().replace(' ', '_')}_{datetime.now().timestamp()}"

    return FoodItem(
        item_id=item_id,
        name=name,
        category=FoodCategory.BAKERY,
        quantity=Decimal(quantity),
        unit="count",
        storage_requirement=StorageRequirement.ROOM_TEMP,
        expiration_date=datetime.now() + timedelta(days=days_until_expiration),
        source_org_id=source_org_id
    )
