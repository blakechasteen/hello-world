"""Inventory models for pantry, fridge, and leftover tracking"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Optional
from enum import Enum


class InventoryLocation(Enum):
    """Where an item is stored"""
    PANTRY = "pantry"
    FRIDGE = "fridge"
    FREEZER = "freezer"
    COUNTER = "counter"


class InventoryUnit(Enum):
    """Measurement units for inventory"""
    ITEM = "item"  # whole items (e.g., 3 onions)
    LB = "lb"
    OZ = "oz"
    CUP = "cup"
    TBSP = "tbsp"
    TSP = "tsp"
    PACKAGE = "package"
    BUNCH = "bunch"


class WasteReason(Enum):
    """Reasons for wasting food"""
    EXPIRED = "expired"  # Past expiration date
    SPOILED = "spoiled"  # Went bad before expiration
    UNUSED = "unused"  # Never used, forgot about it
    OVERCOOKED = "overcooked"  # Cooking mistake
    EXCESS = "excess"  # Made too much
    FREEZER_BURN = "freezer_burn"  # Freezer damage
    OTHER = "other"  # Other reasons


class InventorySource(Enum):
    """Where an item was obtained"""
    STORE = "store"  # Regular grocery store
    GARDEN = "garden"  # Home garden harvest
    LOCAL_FARM = "local_farm"  # Local farm purchase
    FARMERS_MARKET = "farmers_market"  # Farmers market
    CSA = "csa"  # Community Supported Agriculture
    GIFT = "gift"  # Gift from friend/neighbor
    FORAGING = "foraging"  # Wild foraging


class Season(Enum):
    """Seasons for produce availability"""
    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"
    WINTER = "winter"
    YEAR_ROUND = "year_round"


@dataclass
class InventoryItem:
    """
    An item in the pantry, fridge, or freezer

    Attributes:
        ingredient_id: Links to ingredient in ingredients.json
        quantity: Amount available
        unit: Measurement unit
        location: Where it's stored
        source: Where the item came from (store, garden, farm, etc.)
        purchased_date: When the item was purchased (for mental freshness tracking)
        expiration_date: Optional expiration date
        opened_date: When the item was opened (for tracking freshness)
        cost: Optional cost of the item (for budget tracking)
        source_details: Additional source info (e.g., "Smith Family Farm", "backyard tomatoes")
        notes: Additional notes (e.g., "half used", "opened 2 days ago")
    """

    ingredient_id: str
    quantity: float
    unit: InventoryUnit
    location: InventoryLocation
    source: InventorySource = InventorySource.STORE
    purchased_date: Optional[date] = None
    expiration_date: Optional[date] = None
    opened_date: Optional[date] = None
    cost: Optional[float] = None
    source_details: str = ""
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "InventoryItem":
        """Create InventoryItem from JSON dict"""
        return cls(
            ingredient_id=data["ingredient_id"],
            quantity=data["quantity"],
            unit=InventoryUnit(data["unit"]),
            location=InventoryLocation(data["location"]),
            source=InventorySource(data.get("source", "store")),
            purchased_date=date.fromisoformat(data["purchased_date"]) if data.get("purchased_date") else None,
            expiration_date=date.fromisoformat(data["expiration_date"]) if data.get("expiration_date") else None,
            opened_date=date.fromisoformat(data["opened_date"]) if data.get("opened_date") else None,
            cost=data.get("cost"),
            source_details=data.get("source_details", ""),
            notes=data.get("notes", ""),
        )

    def to_dict(self) -> dict:
        """Convert to JSON dict"""
        return {
            "ingredient_id": self.ingredient_id,
            "quantity": self.quantity,
            "unit": self.unit.value,
            "location": self.location.value,
            "source": self.source.value,
            "purchased_date": self.purchased_date.isoformat() if self.purchased_date else None,
            "expiration_date": self.expiration_date.isoformat() if self.expiration_date else None,
            "opened_date": self.opened_date.isoformat() if self.opened_date else None,
            "cost": self.cost,
            "source_details": self.source_details,
            "notes": self.notes,
        }

    def is_expired(self, as_of: Optional[date] = None) -> bool:
        """Check if item is expired"""
        if not self.expiration_date:
            return False
        check_date = as_of or date.today()
        return self.expiration_date < check_date

    def days_until_expiration(self, as_of: Optional[date] = None) -> Optional[int]:
        """Calculate days until expiration (negative if expired)"""
        if not self.expiration_date:
            return None
        check_date = as_of or date.today()
        return (self.expiration_date - check_date).days

    def days_since_purchase(self, as_of: Optional[date] = None) -> Optional[int]:
        """Calculate days since purchase (for mental freshness tracking)"""
        if not self.purchased_date:
            return None
        check_date = as_of or date.today()
        return (check_date - self.purchased_date).days


@dataclass
class Leftover:
    """
    Leftover food from a meal or recipe

    Attributes:
        recipe_id: Which recipe this came from
        description: What it is (e.g., "Mashed Potatoes")
        quantity: Amount remaining
        unit: Measurement unit
        created_date: When the leftover was created
        expiration_date: When it should be eaten by
        location: Where it's stored
        consumed: Whether it's been eaten
        notes: Additional notes
    """

    recipe_id: str
    description: str
    quantity: float
    unit: InventoryUnit
    created_date: date
    expiration_date: date
    location: InventoryLocation = InventoryLocation.FRIDGE
    consumed: bool = False
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "Leftover":
        """Create Leftover from JSON dict"""
        return cls(
            recipe_id=data["recipe_id"],
            description=data["description"],
            quantity=data["quantity"],
            unit=InventoryUnit(data["unit"]),
            created_date=date.fromisoformat(data["created_date"]),
            expiration_date=date.fromisoformat(data["expiration_date"]),
            location=InventoryLocation(data.get("location", "fridge")),
            consumed=data.get("consumed", False),
            notes=data.get("notes", ""),
        )

    def to_dict(self) -> dict:
        """Convert to JSON dict"""
        return {
            "recipe_id": self.recipe_id,
            "description": self.description,
            "quantity": self.quantity,
            "unit": self.unit.value,
            "created_date": self.created_date.isoformat(),
            "expiration_date": self.expiration_date.isoformat(),
            "location": self.location.value,
            "consumed": self.consumed,
            "notes": self.notes,
        }

    def is_expired(self, as_of: Optional[date] = None) -> bool:
        """Check if leftover is expired"""
        check_date = as_of or date.today()
        return self.expiration_date < check_date

    def days_remaining(self, as_of: Optional[date] = None) -> int:
        """Days until expiration (negative if expired)"""
        check_date = as_of or date.today()
        return (self.expiration_date - check_date).days


@dataclass
class WastedItem:
    """
    Track food waste to learn shelf life patterns and track costs

    Attributes:
        ingredient_id: Links to ingredient in ingredients.json
        description: What was wasted (e.g., "Spoiled milk")
        quantity_wasted: Amount that was thrown away
        unit: Measurement unit
        purchased_date: When the item was originally purchased
        wasted_date: When the item was discarded
        reason: Why it was wasted (expired, spoiled, unused, etc.)
        cost: Original cost of the item (for financial tracking)
        expected_expiration: What the expiration date was
        notes: Additional notes
    """

    ingredient_id: str
    description: str
    quantity_wasted: float
    unit: InventoryUnit
    wasted_date: date
    reason: WasteReason
    purchased_date: Optional[date] = None
    cost: Optional[float] = None
    expected_expiration: Optional[date] = None
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "WastedItem":
        """Create WastedItem from JSON dict"""
        return cls(
            ingredient_id=data["ingredient_id"],
            description=data["description"],
            quantity_wasted=data["quantity_wasted"],
            unit=InventoryUnit(data["unit"]),
            wasted_date=date.fromisoformat(data["wasted_date"]),
            reason=WasteReason(data["reason"]),
            purchased_date=date.fromisoformat(data["purchased_date"]) if data.get("purchased_date") else None,
            cost=data.get("cost"),
            expected_expiration=date.fromisoformat(data["expected_expiration"]) if data.get("expected_expiration") else None,
            notes=data.get("notes", ""),
        )

    def to_dict(self) -> dict:
        """Convert to JSON dict"""
        return {
            "ingredient_id": self.ingredient_id,
            "description": self.description,
            "quantity_wasted": self.quantity_wasted,
            "unit": self.unit.value,
            "wasted_date": self.wasted_date.isoformat(),
            "reason": self.reason.value,
            "purchased_date": self.purchased_date.isoformat() if self.purchased_date else None,
            "cost": self.cost,
            "expected_expiration": self.expected_expiration.isoformat() if self.expected_expiration else None,
            "notes": self.notes,
        }

    def actual_shelf_life_days(self) -> Optional[int]:
        """Calculate actual shelf life (how long item lasted from purchase to waste)"""
        if not self.purchased_date:
            return None
        return (self.wasted_date - self.purchased_date).days

    def expected_shelf_life_days(self) -> Optional[int]:
        """Calculate expected shelf life (from purchase to expected expiration)"""
        if not self.purchased_date or not self.expected_expiration:
            return None
        return (self.expected_expiration - self.purchased_date).days

    def days_early_waste(self) -> Optional[int]:
        """How many days before expected expiration was item wasted (negative = wasted after expiration)"""
        if not self.expected_expiration:
            return None
        return (self.expected_expiration - self.wasted_date).days


@dataclass
class GardenHarvest:
    """
    Track garden harvests for cost savings and seasonal planning

    Attributes:
        ingredient_id: Links to ingredient in ingredients.json
        quantity: Amount harvested
        unit: Measurement unit
        harvest_date: When the item was harvested
        garden_location: Where it was grown (e.g., "backyard", "community garden")
        season: Season of harvest
        estimated_value: Estimated retail value (cost savings)
        notes: Additional notes (e.g., "first tomatoes of season")
    """

    ingredient_id: str
    quantity: float
    unit: InventoryUnit
    harvest_date: date
    garden_location: str
    season: Season
    estimated_value: Optional[float] = None
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "GardenHarvest":
        """Create GardenHarvest from JSON dict"""
        return cls(
            ingredient_id=data["ingredient_id"],
            quantity=data["quantity"],
            unit=InventoryUnit(data["unit"]),
            harvest_date=date.fromisoformat(data["harvest_date"]),
            garden_location=data["garden_location"],
            season=Season(data["season"]),
            estimated_value=data.get("estimated_value"),
            notes=data.get("notes", ""),
        )

    def to_dict(self) -> dict:
        """Convert to JSON dict"""
        return {
            "ingredient_id": self.ingredient_id,
            "quantity": self.quantity,
            "unit": self.unit.value,
            "harvest_date": self.harvest_date.isoformat(),
            "garden_location": self.garden_location,
            "season": self.season.value,
            "estimated_value": self.estimated_value,
            "notes": self.notes,
        }
