#!/usr/bin/env python3
"""
Organization Models for Community Food Sharing

Represents different types of organizations in the food sharing network:
- Food Banks / Pantries
- Restaurants
- Grocery Stores
- Churches / Community Centers
- Schools
- Households (from Phase 1)
"""

from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from typing import List, Dict, Optional, Set
from decimal import Decimal


class OrganizationType(Enum):
    """Types of organizations in the network"""
    FOOD_BANK = "food_bank"
    RESTAURANT = "restaurant"
    GROCERY_STORE = "grocery_store"
    CHURCH = "church"
    SCHOOL = "school"
    COMMUNITY_CENTER = "community_center"
    HOUSEHOLD = "household"
    FARM = "farm"


class CertificationType(Enum):
    """Food safety and handling certifications"""
    FOOD_HANDLERS = "food_handlers"
    HACCP = "haccp"
    FOOD_SAFETY_MANAGER = "food_safety_manager"
    SERVSAFE = "servsafe"
    ORGANIC = "organic"
    NON_GMO = "non_gmo"
    KOSHER = "kosher"
    HALAL = "halal"


class DonationCapability(Enum):
    """What an organization can donate"""
    PREPARED_FOOD = "prepared_food"
    FRESH_PRODUCE = "fresh_produce"
    PACKAGED_GOODS = "packaged_goods"
    FROZEN_ITEMS = "frozen_items"
    BAKED_GOODS = "baked_goods"
    DAIRY = "dairy"
    MEAT = "meat"
    BEVERAGES = "beverages"


@dataclass
class OperatingHours:
    """Operating hours for an organization"""
    day_of_week: int  # 0=Monday, 6=Sunday
    open_time: time
    close_time: time
    accepts_donations: bool = True
    accepts_pickups: bool = True

    def is_open_at(self, check_time: datetime) -> bool:
        """Check if organization is open at given time"""
        if check_time.weekday() != self.day_of_week:
            return False

        current_time = check_time.time()
        return self.open_time <= current_time <= self.close_time


@dataclass
class Location:
    """Geographic location"""
    address: str
    city: str
    state: str
    zip_code: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    def get_full_address(self) -> str:
        """Get formatted full address"""
        return f"{self.address}, {self.city}, {self.state} {self.zip_code}"


@dataclass
class Organization:
    """
    Base organization in the food sharing network.

    All organizations can participate in food sharing, either as:
    - Donors (have surplus food)
    - Recipients (need food)
    - Both
    """
    org_id: str
    name: str
    org_type: OrganizationType
    location: Location

    # Contact information
    contact_name: str
    contact_email: str
    contact_phone: str

    # Capabilities
    can_donate: bool = True
    can_receive: bool = True
    donation_capabilities: List[DonationCapability] = field(default_factory=list)

    # Operating hours
    operating_hours: List[OperatingHours] = field(default_factory=list)

    # Certifications and compliance
    certifications: List[CertificationType] = field(default_factory=list)
    last_inspection_date: Optional[datetime] = None

    # Logistics
    has_refrigeration: bool = False
    has_freezer: bool = False
    delivery_radius_miles: float = 5.0
    can_pickup: bool = True
    can_deliver: bool = False

    # Network metrics
    total_donations_made: int = 0
    total_donations_received: int = 0
    total_weight_donated_lbs: Decimal = field(default_factory=lambda: Decimal("0.0"))
    total_weight_received_lbs: Decimal = field(default_factory=lambda: Decimal("0.0"))
    reputation_score: float = 5.0  # 0-5 stars

    # Status
    active: bool = True
    verified: bool = False
    joined_date: datetime = field(default_factory=datetime.now)

    def is_open_now(self) -> bool:
        """Check if organization is currently open"""
        now = datetime.now()
        today = now.weekday()

        for hours in self.operating_hours:
            if hours.day_of_week == today and hours.is_open_at(now):
                return True

        return False

    def can_accept_donation_type(self, capability: DonationCapability) -> bool:
        """Check if organization can handle specific donation type"""
        if not self.can_receive:
            return False

        # Food banks typically accept everything
        if self.org_type == OrganizationType.FOOD_BANK:
            return True

        return capability in self.donation_capabilities

    def get_distance_to(self, other_location: Location) -> Optional[float]:
        """
        Calculate distance to another location in miles.

        Returns None if coordinates not available.
        """
        if not all([
            self.location.latitude,
            self.location.longitude,
            other_location.latitude,
            other_location.longitude
        ]):
            return None

        # Haversine formula
        from math import radians, sin, cos, sqrt, atan2

        lat1 = radians(self.location.latitude)
        lon1 = radians(self.location.longitude)
        lat2 = radians(other_location.latitude)
        lon2 = radians(other_location.longitude)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))

        # Earth's radius in miles
        radius = 3959.0

        return radius * c

    def can_deliver_to(self, other_location: Location) -> bool:
        """Check if organization can deliver to location"""
        if not self.can_deliver:
            return False

        distance = self.get_distance_to(other_location)
        if distance is None:
            return False

        return distance <= self.delivery_radius_miles

    def update_reputation(self, rating: float):
        """Update reputation score with new rating"""
        # Weighted average: 80% existing, 20% new rating
        self.reputation_score = (self.reputation_score * 0.8) + (rating * 0.2)
        self.reputation_score = max(0.0, min(5.0, self.reputation_score))

    def to_dict(self) -> dict:
        """Serialize to dictionary"""
        return {
            "org_id": self.org_id,
            "name": self.name,
            "org_type": self.org_type.value,
            "location": {
                "address": self.location.address,
                "city": self.location.city,
                "state": self.location.state,
                "zip_code": self.location.zip_code,
                "latitude": self.location.latitude,
                "longitude": self.location.longitude
            },
            "contact": {
                "name": self.contact_name,
                "email": self.contact_email,
                "phone": self.contact_phone
            },
            "capabilities": {
                "can_donate": self.can_donate,
                "can_receive": self.can_receive,
                "donation_capabilities": [c.value for c in self.donation_capabilities]
            },
            "certifications": [c.value for c in self.certifications],
            "logistics": {
                "has_refrigeration": self.has_refrigeration,
                "has_freezer": self.has_freezer,
                "delivery_radius_miles": float(self.delivery_radius_miles),
                "can_pickup": self.can_pickup,
                "can_deliver": self.can_deliver
            },
            "metrics": {
                "total_donations_made": self.total_donations_made,
                "total_donations_received": self.total_donations_received,
                "total_weight_donated_lbs": float(self.total_weight_donated_lbs),
                "total_weight_received_lbs": float(self.total_weight_received_lbs),
                "reputation_score": self.reputation_score
            },
            "status": {
                "active": self.active,
                "verified": self.verified,
                "is_open_now": self.is_open_now()
            }
        }


@dataclass
class FoodBank(Organization):
    """
    Food bank or pantry.

    Typically receives donations from many sources and distributes
    to individuals and families.
    """
    # Capacity
    max_families_per_day: int = 50
    current_inventory_lbs: Decimal = field(default_factory=lambda: Decimal("0.0"))
    max_inventory_lbs: Decimal = field(default_factory=lambda: Decimal("10000.0"))

    # Service area
    zip_codes_served: List[str] = field(default_factory=list)
    households_served: int = 0

    # Special programs
    senior_program: bool = False
    snap_enrollment: bool = False
    nutrition_education: bool = False

    def __post_init__(self):
        """Ensure food bank defaults"""
        self.org_type = OrganizationType.FOOD_BANK
        self.can_receive = True
        self.can_donate = True  # Can donate to other food banks

        # Food banks typically have refrigeration
        self.has_refrigeration = True
        self.has_freezer = True

    def get_capacity_percent(self) -> float:
        """Get current capacity as percentage"""
        if self.max_inventory_lbs == 0:
            return 0.0
        return float(self.current_inventory_lbs / self.max_inventory_lbs * 100)

    def can_accept_more(self, weight_lbs: Decimal) -> bool:
        """Check if food bank can accept more donations"""
        if not self.active:
            return False

        new_total = self.current_inventory_lbs + weight_lbs
        return new_total <= self.max_inventory_lbs

    def serves_zip_code(self, zip_code: str) -> bool:
        """Check if food bank serves a zip code"""
        if not self.zip_codes_served:
            return True  # Serves everyone if not specified

        return zip_code in self.zip_codes_served


@dataclass
class Restaurant(Organization):
    """
    Restaurant with potential surplus prepared food.

    Can donate leftover prepared meals at end of service.
    """
    # Restaurant details
    cuisine_type: str = "American"
    seating_capacity: int = 50
    avg_daily_covers: int = 30

    # Donation patterns
    typical_surplus_lbs: Decimal = field(default_factory=lambda: Decimal("0.0"))
    surplus_days: List[int] = field(default_factory=list)  # Days of week with surplus
    surplus_time: Optional[time] = None  # Time when surplus available

    # Food safety
    food_safety_certified: bool = False
    last_health_inspection: Optional[datetime] = None
    health_score: Optional[int] = None

    def __post_init__(self):
        """Ensure restaurant defaults"""
        self.org_type = OrganizationType.RESTAURANT
        self.can_donate = True
        self.can_receive = False  # Restaurants don't typically receive

        # Restaurants donate prepared food
        if DonationCapability.PREPARED_FOOD not in self.donation_capabilities:
            self.donation_capabilities.append(DonationCapability.PREPARED_FOOD)

    def has_surplus_today(self) -> bool:
        """Check if restaurant typically has surplus today"""
        today = datetime.now().weekday()

        if not self.surplus_days:
            return True  # Assume surplus possible any day if not specified

        return today in self.surplus_days

    def estimate_surplus_lbs(self) -> Decimal:
        """Estimate surplus for today based on patterns"""
        if not self.has_surplus_today():
            return Decimal("0.0")

        return self.typical_surplus_lbs


@dataclass
class GroceryStore(Organization):
    """
    Grocery store with potential surplus food.

    Can donate:
    - Cosmetically imperfect produce
    - Near-expiration packaged goods
    - Bakery day-old items
    """
    # Store details
    store_chain: Optional[str] = None
    square_footage: int = 10000

    # Departments
    has_bakery: bool = True
    has_deli: bool = True
    has_produce: bool = True
    has_meat_counter: bool = True

    # Donation program
    corporate_donation_program: bool = False
    daily_donation_window: Optional[tuple[time, time]] = None

    # Typical donations
    avg_daily_donation_lbs: Decimal = field(default_factory=lambda: Decimal("0.0"))

    def __post_init__(self):
        """Ensure grocery store defaults"""
        self.org_type = OrganizationType.GROCERY_STORE
        self.can_donate = True
        self.can_receive = False  # Stores don't receive

        # Stores have extensive refrigeration
        self.has_refrigeration = True
        self.has_freezer = True

        # Add typical donation capabilities
        if self.has_produce:
            self.donation_capabilities.append(DonationCapability.FRESH_PRODUCE)
        if self.has_bakery:
            self.donation_capabilities.append(DonationCapability.BAKED_GOODS)
        if self.has_deli:
            self.donation_capabilities.append(DonationCapability.PREPARED_FOOD)
        if self.has_meat_counter:
            self.donation_capabilities.append(DonationCapability.MEAT)

    def is_donation_window_now(self) -> bool:
        """Check if currently in donation window"""
        if not self.daily_donation_window:
            return self.is_open_now()

        now = datetime.now().time()
        start, end = self.daily_donation_window

        return start <= now <= end


@dataclass
class Church(Organization):
    """
    Church or religious community center.

    Often serves as both donation collection point and distribution center.
    """
    # Church details
    denomination: Optional[str] = None
    congregation_size: int = 100

    # Programs
    food_pantry_program: bool = False
    meal_program: bool = False
    meal_days: List[int] = field(default_factory=list)  # Days meals are served

    # Facilities
    has_commercial_kitchen: bool = False
    has_food_pantry: bool = False

    def __post_init__(self):
        """Ensure church defaults"""
        self.org_type = OrganizationType.CHURCH

        # Churches typically both receive and redistribute
        self.can_receive = True
        self.can_donate = self.meal_program  # Can donate if they prepare meals

    def serves_meals_today(self) -> bool:
        """Check if church serves meals today"""
        if not self.meal_program:
            return False

        today = datetime.now().weekday()
        return today in self.meal_days


# Factory functions for common organization types

def create_food_bank(
    name: str,
    address: str,
    city: str,
    state: str,
    zip_code: str,
    contact_name: str,
    contact_email: str,
    contact_phone: str,
    max_families_per_day: int = 50
) -> FoodBank:
    """Create a food bank organization"""
    location = Location(
        address=address,
        city=city,
        state=state,
        zip_code=zip_code
    )

    org_id = f"fb_{name.lower().replace(' ', '_')}"

    return FoodBank(
        org_id=org_id,
        name=name,
        org_type=OrganizationType.FOOD_BANK,
        location=location,
        contact_name=contact_name,
        contact_email=contact_email,
        contact_phone=contact_phone,
        max_families_per_day=max_families_per_day
    )


def create_restaurant(
    name: str,
    cuisine_type: str,
    address: str,
    city: str,
    state: str,
    zip_code: str,
    contact_name: str,
    contact_email: str,
    contact_phone: str
) -> Restaurant:
    """Create a restaurant organization"""
    location = Location(
        address=address,
        city=city,
        state=state,
        zip_code=zip_code
    )

    org_id = f"rest_{name.lower().replace(' ', '_')}"

    return Restaurant(
        org_id=org_id,
        name=name,
        org_type=OrganizationType.RESTAURANT,
        location=location,
        contact_name=contact_name,
        contact_email=contact_email,
        contact_phone=contact_phone,
        cuisine_type=cuisine_type
    )


def create_grocery_store(
    name: str,
    chain: Optional[str],
    address: str,
    city: str,
    state: str,
    zip_code: str,
    contact_name: str,
    contact_email: str,
    contact_phone: str
) -> GroceryStore:
    """Create a grocery store organization"""
    location = Location(
        address=address,
        city=city,
        state=state,
        zip_code=zip_code
    )

    org_id = f"store_{name.lower().replace(' ', '_')}"

    return GroceryStore(
        org_id=org_id,
        name=name,
        org_type=OrganizationType.GROCERY_STORE,
        location=location,
        contact_name=contact_name,
        contact_email=contact_email,
        contact_phone=contact_phone,
        store_chain=chain
    )
