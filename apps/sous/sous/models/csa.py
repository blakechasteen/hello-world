#!/usr/bin/env python3
"""
CSA (Community Supported Agriculture) Models for Phase 3

Represents CSA subscriptions, shares, and delivery schedules that connect
farms with consumers through subscription-based food sharing.
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import List, Dict, Optional, Set
from decimal import Decimal


class CSAShareSize(Enum):
    """Size of CSA share"""
    MINI = "mini"  # 1-2 people
    SMALL = "small"  # 2-3 people
    REGULAR = "regular"  # 3-4 people
    LARGE = "large"  # 4-6 people
    FAMILY = "family"  # 6+ people


class CSADeliveryFrequency(Enum):
    """How often shares are delivered"""
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"


class CSASeasonType(Enum):
    """CSA season types"""
    FULL_YEAR = "full_year"  # 52 weeks
    SPRING_SUMMER = "spring_summer"  # 20-24 weeks
    FALL_WINTER = "fall_winter"  # 20-24 weeks
    SPRING = "spring"  # 10-12 weeks
    SUMMER = "summer"  # 10-12 weeks
    FALL = "fall"  # 10-12 weeks
    WINTER = "winter"  # 10-12 weeks


class DeliveryMethod(Enum):
    """How shares are delivered"""
    FARM_PICKUP = "farm_pickup"
    DROPOFF_LOCATION = "dropoff_location"
    HOME_DELIVERY = "home_delivery"
    FARMERS_MARKET = "farmers_market"


class SubscriptionStatus(Enum):
    """Status of CSA subscription"""
    PENDING = "pending"  # Awaiting payment
    ACTIVE = "active"  # Currently receiving shares
    PAUSED = "paused"  # Temporarily suspended
    CANCELLED = "cancelled"  # Cancelled by member
    COMPLETED = "completed"  # Season finished


class PaymentSchedule(Enum):
    """Payment schedule for subscription"""
    FULL_UPFRONT = "full_upfront"  # Pay entire season upfront
    MONTHLY = "monthly"  # Monthly installments
    PER_DELIVERY = "per_delivery"  # Pay per share


@dataclass
class CSAShareItem:
    """Individual item in a CSA share"""
    item_name: str
    quantity: Decimal
    unit: str  # lbs, bunches, heads, etc.
    estimated_value: Decimal = field(default_factory=lambda: Decimal("0.0"))
    source_farm: Optional[str] = None  # farm_id if from specific farm


@dataclass
class CSAShare:
    """
    A single CSA share delivery.

    Represents what a member receives in one delivery.
    """
    share_id: str
    subscription_id: str
    farm_id: str

    # Delivery info
    delivery_date: date
    delivered: bool = False
    delivery_confirmed_at: Optional[datetime] = None

    # Contents
    items: List[CSAShareItem] = field(default_factory=list)
    total_weight_lbs: Decimal = field(default_factory=lambda: Decimal("0.0"))
    estimated_total_value: Decimal = field(default_factory=lambda: Decimal("0.0"))

    # Communication
    farmer_note: str = ""  # Message from farmer about this week's share
    recipe_suggestions: List[str] = field(default_factory=list)
    storage_tips: str = ""

    # Feedback
    member_rating: Optional[int] = None  # 1-5 stars
    member_feedback: str = ""

    def mark_delivered(self):
        """Mark share as delivered"""
        self.delivered = True
        self.delivery_confirmed_at = datetime.now()

    def calculate_total_value(self) -> Decimal:
        """Calculate total value of items in share"""
        total = sum(item.estimated_value for item in self.items)
        self.estimated_total_value = total
        return total

    def calculate_total_weight(self) -> Decimal:
        """Calculate total weight of items in share"""
        total = sum(item.quantity for item in self.items if item.unit in ["lbs", "lb"])
        self.total_weight_lbs = total
        return total


@dataclass
class DeliveryLocation:
    """Location for share pickup/delivery"""
    location_id: str
    name: str  # e.g., "Downtown Community Center"
    address: str
    city: str
    state: str
    zip_code: str

    # Pickup details
    pickup_day: str  # "Monday", "Tuesday", etc.
    pickup_time_start: str  # "16:00"
    pickup_time_end: str  # "19:00"

    # Capacity
    max_members: int = 50
    current_members: int = 0

    # Instructions
    parking_instructions: str = ""
    access_instructions: str = ""

    def is_full(self) -> bool:
        """Check if location is at capacity"""
        return self.current_members >= self.max_members

    def has_space_for(self, count: int = 1) -> bool:
        """Check if location has space for N more members"""
        return (self.current_members + count) <= self.max_members


@dataclass
class DeliverySchedule:
    """
    Delivery schedule for CSA shares.

    Defines when and where shares are delivered throughout the season.
    """
    schedule_id: str
    farm_id: str
    season_name: str  # e.g., "Spring 2024"

    # Schedule details
    start_date: date
    end_date: date
    frequency: CSADeliveryFrequency
    delivery_method: DeliveryMethod

    # Delivery dates (calculated based on frequency)
    delivery_dates: List[date] = field(default_factory=list)

    # Locations
    pickup_locations: List[DeliveryLocation] = field(default_factory=list)

    # Holidays/skips
    skip_dates: Set[date] = field(default_factory=set)
    holiday_closures: Set[date] = field(default_factory=set)

    def generate_delivery_dates(self):
        """Generate all delivery dates for the season"""
        self.delivery_dates = []
        current_date = self.start_date

        # Calculate interval based on frequency
        if self.frequency == CSADeliveryFrequency.WEEKLY:
            interval = timedelta(weeks=1)
        elif self.frequency == CSADeliveryFrequency.BIWEEKLY:
            interval = timedelta(weeks=2)
        elif self.frequency == CSADeliveryFrequency.MONTHLY:
            interval = timedelta(days=30)
        else:
            interval = timedelta(weeks=1)

        while current_date <= self.end_date:
            # Skip dates and holidays
            if current_date not in self.skip_dates and current_date not in self.holiday_closures:
                self.delivery_dates.append(current_date)
            current_date += interval

    def get_next_delivery_date(self) -> Optional[date]:
        """Get next scheduled delivery date"""
        today = date.today()
        for delivery_date in self.delivery_dates:
            if delivery_date > today:
                return delivery_date
        return None

    def get_total_deliveries(self) -> int:
        """Get total number of deliveries in season"""
        return len(self.delivery_dates)

    def add_skip_date(self, skip_date: date, reason: str = ""):
        """Add a date to skip (e.g., farm vacation)"""
        self.skip_dates.add(skip_date)
        # Regenerate delivery dates
        self.generate_delivery_dates()

    def add_pickup_location(self, location: DeliveryLocation):
        """Add a pickup location"""
        self.pickup_locations.append(location)


@dataclass
class CSASubscription:
    """
    CSA subscription linking a member to a farm.

    Represents the ongoing relationship between a CSA member and their farm.
    """
    subscription_id: str
    member_id: str  # User ID from Phase 1
    farm_id: str

    # Subscription details
    season_type: CSASeasonType
    share_size: CSAShareSize
    delivery_frequency: CSADeliveryFrequency
    delivery_method: DeliveryMethod

    # Timing
    start_date: date
    end_date: date
    signed_up_at: datetime = field(default_factory=datetime.now)

    # Status
    status: SubscriptionStatus = SubscriptionStatus.PENDING

    # Delivery
    delivery_schedule_id: Optional[str] = None
    pickup_location_id: Optional[str] = None
    home_delivery_address: Optional[str] = None

    # Payment
    payment_schedule: PaymentSchedule = PaymentSchedule.FULL_UPFRONT
    total_cost: Decimal = field(default_factory=lambda: Decimal("0.0"))
    amount_paid: Decimal = field(default_factory=lambda: Decimal("0.0"))
    payment_status: str = "pending"  # pending, partial, paid

    # Shares
    total_shares_expected: int = 0
    shares_received: int = 0
    shares: List[str] = field(default_factory=list)  # List of share_ids

    # Preferences
    exclude_items: Set[str] = field(default_factory=set)  # Items member doesn't want
    prefer_organic: bool = False

    # Communication
    email_notifications: bool = True
    sms_notifications: bool = False

    # Satisfaction
    overall_rating: Optional[int] = None  # 1-5 stars at end of season
    renewal_likely: Optional[bool] = None
    testimonial: str = ""

    def activate(self):
        """Activate subscription after payment"""
        if self.amount_paid >= self.total_cost:
            self.status = SubscriptionStatus.ACTIVE
            self.payment_status = "paid"

    def pause(self):
        """Temporarily pause subscription"""
        if self.status == SubscriptionStatus.ACTIVE:
            self.status = SubscriptionStatus.PAUSED

    def resume(self):
        """Resume paused subscription"""
        if self.status == SubscriptionStatus.PAUSED:
            self.status = SubscriptionStatus.ACTIVE

    def cancel(self):
        """Cancel subscription"""
        self.status = SubscriptionStatus.CANCELLED

    def complete(self):
        """Mark subscription as completed at end of season"""
        self.status = SubscriptionStatus.COMPLETED

    def record_share_received(self, share_id: str):
        """Record that a share was received"""
        self.shares.append(share_id)
        self.shares_received += 1

    def get_fulfillment_rate(self) -> float:
        """Calculate percentage of shares received"""
        if self.total_shares_expected == 0:
            return 0.0
        return (self.shares_received / self.total_shares_expected) * 100

    def get_cost_per_share(self) -> Decimal:
        """Calculate cost per share"""
        if self.total_shares_expected == 0:
            return Decimal("0.0")
        return self.total_cost / Decimal(str(self.total_shares_expected))

    def is_active(self) -> bool:
        """Check if subscription is currently active"""
        return self.status == SubscriptionStatus.ACTIVE

    def needs_payment(self) -> bool:
        """Check if payment is due"""
        return self.amount_paid < self.total_cost


@dataclass
class CSASeason:
    """
    A CSA season offered by a farm.

    Defines the parameters for a farm's CSA program for a particular season.
    """
    season_id: str
    farm_id: str

    # Season details
    name: str  # e.g., "Spring 2024 CSA"
    season_type: CSASeasonType
    year: int

    # Timing
    signup_opens: date
    signup_closes: date
    season_starts: date
    season_ends: date

    # Offerings
    available_share_sizes: List[CSAShareSize] = field(default_factory=list)
    available_frequencies: List[CSADeliveryFrequency] = field(default_factory=list)
    available_delivery_methods: List[DeliveryMethod] = field(default_factory=list)

    # Pricing
    pricing: Dict[CSAShareSize, Decimal] = field(default_factory=dict)
    # e.g., {CSAShareSize.REGULAR: Decimal("600.00")}

    # Capacity
    max_members: int = 100
    current_members: int = 0
    waitlist_count: int = 0

    # Delivery schedule
    delivery_schedule_id: Optional[str] = None

    # Status
    accepting_signups: bool = True
    active: bool = False
    completed: bool = False

    # Marketing
    description: str = ""
    expected_items: List[str] = field(default_factory=list)
    farm_story: str = ""

    def is_signup_open(self) -> bool:
        """Check if signup period is open"""
        today = date.today()
        return (self.accepting_signups and
                self.signup_opens <= today <= self.signup_closes and
                not self.is_full())

    def is_full(self) -> bool:
        """Check if season is at capacity"""
        return self.current_members >= self.max_members

    def has_space_for(self, count: int = 1) -> bool:
        """Check if season has space for N more members"""
        return (self.current_members + count) <= self.max_members

    def get_price_for_share_size(self, share_size: CSAShareSize) -> Optional[Decimal]:
        """Get price for a share size"""
        return self.pricing.get(share_size)

    def start_season(self):
        """Activate the season"""
        self.active = True
        self.accepting_signups = False

    def end_season(self):
        """Complete the season"""
        self.active = False
        self.completed = True


# ========================================================================
# Factory Functions
# ========================================================================

def create_csa_season(
    farm_id: str,
    name: str,
    season_type: CSASeasonType,
    year: int,
    signup_opens: date,
    season_starts: date,
    season_ends: date,
    max_members: int = 100
) -> CSASeason:
    """Create a CSA season"""
    season_id = f"season_{farm_id}_{year}_{season_type.value}"

    # Default pricing based on share size
    default_pricing = {
        CSAShareSize.MINI: Decimal("300.00"),
        CSAShareSize.SMALL: Decimal("450.00"),
        CSAShareSize.REGULAR: Decimal("600.00"),
        CSAShareSize.LARGE: Decimal("800.00"),
        CSAShareSize.FAMILY: Decimal("1000.00")
    }

    # Signup closes 2 weeks before season starts
    signup_closes = season_starts - timedelta(weeks=2)

    return CSASeason(
        season_id=season_id,
        farm_id=farm_id,
        name=name,
        season_type=season_type,
        year=year,
        signup_opens=signup_opens,
        signup_closes=signup_closes,
        season_starts=season_starts,
        season_ends=season_ends,
        available_share_sizes=[
            CSAShareSize.SMALL,
            CSAShareSize.REGULAR,
            CSAShareSize.LARGE
        ],
        available_frequencies=[
            CSADeliveryFrequency.WEEKLY,
            CSADeliveryFrequency.BIWEEKLY
        ],
        available_delivery_methods=[
            DeliveryMethod.FARM_PICKUP,
            DeliveryMethod.DROPOFF_LOCATION
        ],
        pricing=default_pricing,
        max_members=max_members,
        accepting_signups=True
    )


def create_subscription(
    member_id: str,
    farm_id: str,
    season: CSASeason,
    share_size: CSAShareSize,
    delivery_method: DeliveryMethod,
    pickup_location_id: Optional[str] = None
) -> CSASubscription:
    """Create a CSA subscription"""
    subscription_id = f"sub_{member_id}_{farm_id}_{season.year}"

    # Get pricing from season
    total_cost = season.get_price_for_share_size(share_size) or Decimal("600.00")

    # Calculate expected shares based on frequency
    if season.season_type == CSASeasonType.FULL_YEAR:
        total_shares = 52
    elif season.season_type in [CSASeasonType.SPRING_SUMMER, CSASeasonType.FALL_WINTER]:
        total_shares = 24
    else:
        total_shares = 12

    # Adjust for frequency
    frequency = CSADeliveryFrequency.WEEKLY
    if frequency == CSADeliveryFrequency.BIWEEKLY:
        total_shares = total_shares // 2
    elif frequency == CSADeliveryFrequency.MONTHLY:
        total_shares = total_shares // 4

    return CSASubscription(
        subscription_id=subscription_id,
        member_id=member_id,
        farm_id=farm_id,
        season_type=season.season_type,
        share_size=share_size,
        delivery_frequency=frequency,
        delivery_method=delivery_method,
        start_date=season.season_starts,
        end_date=season.season_ends,
        pickup_location_id=pickup_location_id,
        total_cost=total_cost,
        total_shares_expected=total_shares,
        status=SubscriptionStatus.PENDING
    )


def create_delivery_schedule(
    farm_id: str,
    season_name: str,
    start_date: date,
    end_date: date,
    frequency: CSADeliveryFrequency = CSADeliveryFrequency.WEEKLY,
    delivery_method: DeliveryMethod = DeliveryMethod.FARM_PICKUP
) -> DeliverySchedule:
    """Create a delivery schedule"""
    schedule_id = f"schedule_{farm_id}_{start_date.year}"

    schedule = DeliverySchedule(
        schedule_id=schedule_id,
        farm_id=farm_id,
        season_name=season_name,
        start_date=start_date,
        end_date=end_date,
        frequency=frequency,
        delivery_method=delivery_method
    )

    # Generate delivery dates
    schedule.generate_delivery_dates()

    return schedule


def create_csa_share(
    subscription_id: str,
    farm_id: str,
    delivery_date: date,
    items: List[CSAShareItem],
    farmer_note: str = ""
) -> CSAShare:
    """Create a CSA share"""
    share_id = f"share_{subscription_id}_{delivery_date.isoformat()}"

    share = CSAShare(
        share_id=share_id,
        subscription_id=subscription_id,
        farm_id=farm_id,
        delivery_date=delivery_date,
        items=items,
        farmer_note=farmer_note
    )

    # Calculate totals
    share.calculate_total_value()
    share.calculate_total_weight()

    return share
