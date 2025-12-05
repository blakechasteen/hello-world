#!/usr/bin/env python3
"""
CSA Coordinator Service for Phase 3

Manages Community Supported Agriculture (CSA) programs including:
- Season management
- Subscription management
- Share generation and delivery
- Member communication
- Analytics and retention
"""

from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Set, Tuple
from decimal import Decimal
from collections import defaultdict

from sous.models.farm import Farm
from sous.models.csa import (
    CSASeason, CSASubscription, CSAShare, DeliverySchedule, DeliveryLocation,
    CSAShareItem, CSAShareSize, CSADeliveryFrequency, CSASeasonType,
    DeliveryMethod, SubscriptionStatus, PaymentSchedule,
    create_csa_season, create_subscription, create_delivery_schedule, create_csa_share
)


@dataclass
class CSAAnalytics:
    """Analytics for CSA program"""
    farm_id: str
    season_id: str
    period_start: date
    period_end: date

    # Membership
    total_members: int = 0
    active_members: int = 0
    paused_members: int = 0
    cancelled_members: int = 0

    # Revenue
    total_revenue: Decimal = Decimal("0.0")
    projected_revenue: Decimal = Decimal("0.0")
    average_subscription_value: Decimal = Decimal("0.0")

    # Shares
    total_shares_delivered: int = 0
    shares_scheduled: int = 0
    delivery_completion_rate: float = 0.0

    # Satisfaction
    average_rating: float = 0.0
    renewal_rate: float = 0.0
    feedback_count: int = 0

    # Retention
    member_retention_rate: float = 0.0
    churn_rate: float = 0.0


@dataclass
class SharePlan:
    """Plan for a share delivery"""
    share_date: date
    items: List[CSAShareItem]
    notes: str = ""
    recipes: List[str] = None

    def __post_init__(self):
        if self.recipes is None:
            self.recipes = []


class CSACoordinator:
    """
    CSA program coordination system.

    Manages seasons, subscriptions, shares, and member relationships.
    """

    def __init__(self):
        # Core data structures
        self.farms: Dict[str, Farm] = {}
        self.seasons: Dict[str, CSASeason] = {}
        self.subscriptions: Dict[str, CSASubscription] = {}
        self.shares: Dict[str, CSAShare] = {}
        self.schedules: Dict[str, DeliverySchedule] = {}
        self.locations: Dict[str, DeliveryLocation] = {}

        # Share plans (templates for generating shares)
        self.share_plans: Dict[str, List[SharePlan]] = defaultdict(list)

        # Indexes
        self.seasons_by_farm: Dict[str, List[str]] = defaultdict(list)
        self.subscriptions_by_member: Dict[str, List[str]] = defaultdict(list)
        self.subscriptions_by_season: Dict[str, List[str]] = defaultdict(list)
        self.shares_by_subscription: Dict[str, List[str]] = defaultdict(list)

    # ========================================================================
    # Farm Registration
    # ========================================================================

    def register_farm(self, farm: Farm):
        """Register a farm offering CSA"""
        if not farm.offers_csa:
            raise ValueError(f"Farm {farm.name} does not offer CSA")

        self.farms[farm.org_id] = farm

    def get_farm(self, farm_id: str) -> Optional[Farm]:
        """Get farm by ID"""
        return self.farms.get(farm_id)

    # ========================================================================
    # Season Management
    # ========================================================================

    def create_season(
        self,
        farm_id: str,
        name: str,
        season_type: CSASeasonType,
        year: int,
        signup_opens: date,
        season_starts: date,
        season_ends: date,
        max_members: int = 100
    ) -> CSASeason:
        """Create a new CSA season"""
        farm = self.get_farm(farm_id)
        if not farm:
            raise ValueError(f"Farm {farm_id} not found")

        season = create_csa_season(
            farm_id, name, season_type, year,
            signup_opens, season_starts, season_ends, max_members
        )

        # Create delivery schedule
        schedule = create_delivery_schedule(
            farm_id, name, season_starts, season_ends
        )

        season.delivery_schedule_id = schedule.schedule_id
        self.schedules[schedule.schedule_id] = schedule

        # Store season
        self.seasons[season.season_id] = season
        self.seasons_by_farm[farm_id].append(season.season_id)

        return season

    def get_season(self, season_id: str) -> Optional[CSASeason]:
        """Get season by ID"""
        return self.seasons.get(season_id)

    def get_farm_seasons(self, farm_id: str, active_only: bool = False) -> List[CSASeason]:
        """Get all seasons for a farm"""
        season_ids = self.seasons_by_farm.get(farm_id, [])
        seasons = [self.seasons[sid] for sid in season_ids if sid in self.seasons]

        if active_only:
            seasons = [s for s in seasons if s.active]

        return seasons

    def get_available_seasons(self) -> List[CSASeason]:
        """Get all seasons currently accepting signups"""
        all_seasons = list(self.seasons.values())
        return [s for s in all_seasons if s.is_signup_open()]

    def start_season(self, season_id: str):
        """Activate a season"""
        season = self.get_season(season_id)
        if not season:
            raise ValueError(f"Season {season_id} not found")

        season.start_season()

        # Generate shares for all subscriptions
        subscriptions = self.get_season_subscriptions(season_id, active_only=True)
        for subscription in subscriptions:
            self.generate_season_shares(subscription.subscription_id)

    def end_season(self, season_id: str):
        """Complete a season"""
        season = self.get_season(season_id)
        if not season:
            raise ValueError(f"Season {season_id} not found")

        season.end_season()

        # Complete all active subscriptions
        subscriptions = self.get_season_subscriptions(season_id, active_only=True)
        for subscription in subscriptions:
            subscription.complete()

    # ========================================================================
    # Subscription Management
    # ========================================================================

    def create_subscription(
        self,
        member_id: str,
        season_id: str,
        share_size: CSAShareSize,
        delivery_method: DeliveryMethod,
        pickup_location_id: Optional[str] = None,
        delivery_address: Optional[str] = None
    ) -> CSASubscription:
        """Create a new CSA subscription"""
        season = self.get_season(season_id)
        if not season:
            raise ValueError(f"Season {season_id} not found")

        if not season.is_signup_open():
            raise ValueError(f"Season {season.name} is not accepting signups")

        # Create subscription
        subscription = create_subscription(
            member_id, season.farm_id, season,
            share_size, delivery_method, pickup_location_id
        )

        subscription.delivery_schedule_id = season.delivery_schedule_id

        if delivery_method == DeliveryMethod.HOME_DELIVERY:
            subscription.home_delivery_address = delivery_address

        # Store subscription
        self.subscriptions[subscription.subscription_id] = subscription
        self.subscriptions_by_member[member_id].append(subscription.subscription_id)
        self.subscriptions_by_season[season_id].append(subscription.subscription_id)

        # Increment season member count
        season.current_members += 1

        return subscription

    def get_subscription(self, subscription_id: str) -> Optional[CSASubscription]:
        """Get subscription by ID"""
        return self.subscriptions.get(subscription_id)

    def get_member_subscriptions(
        self,
        member_id: str,
        active_only: bool = False
    ) -> List[CSASubscription]:
        """Get all subscriptions for a member"""
        subscription_ids = self.subscriptions_by_member.get(member_id, [])
        subscriptions = [
            self.subscriptions[sid] for sid in subscription_ids
            if sid in self.subscriptions
        ]

        if active_only:
            subscriptions = [s for s in subscriptions if s.is_active()]

        return subscriptions

    def get_season_subscriptions(
        self,
        season_id: str,
        active_only: bool = False
    ) -> List[CSASubscription]:
        """Get all subscriptions for a season"""
        subscription_ids = self.subscriptions_by_season.get(season_id, [])
        subscriptions = [
            self.subscriptions[sid] for sid in subscription_ids
            if sid in self.subscriptions
        ]

        if active_only:
            subscriptions = [s for s in subscriptions if s.is_active()]

        return subscriptions

    def activate_subscription(self, subscription_id: str, payment_amount: Decimal):
        """Activate subscription after payment"""
        subscription = self.get_subscription(subscription_id)
        if not subscription:
            raise ValueError(f"Subscription {subscription_id} not found")

        subscription.amount_paid += payment_amount

        if subscription.amount_paid >= subscription.total_cost:
            subscription.activate()

    def pause_subscription(self, subscription_id: str, weeks: int = 1):
        """Temporarily pause subscription"""
        subscription = self.get_subscription(subscription_id)
        if not subscription:
            raise ValueError(f"Subscription {subscription_id} not found")

        subscription.pause()

        # Mark upcoming shares as skipped
        # In production, would calculate which specific shares to skip

    def resume_subscription(self, subscription_id: str):
        """Resume paused subscription"""
        subscription = self.get_subscription(subscription_id)
        if not subscription:
            raise ValueError(f"Subscription {subscription_id} not found")

        subscription.resume()

    def cancel_subscription(self, subscription_id: str, reason: str = ""):
        """Cancel subscription"""
        subscription = self.get_subscription(subscription_id)
        if not subscription:
            raise ValueError(f"Subscription {subscription_id} not found")

        subscription.cancel()

        # Adjust season member count
        season = self.get_season(f"season_{subscription.farm_id}_{subscription.start_date.year}_{subscription.season_type.value}")
        if season:
            season.current_members -= 1

    # ========================================================================
    # Share Generation and Delivery
    # ========================================================================

    def add_share_plan(self, season_id: str, share_date: date, items: List[CSAShareItem], notes: str = ""):
        """Add a share plan for a specific date"""
        plan = SharePlan(share_date=share_date, items=items, notes=notes)
        self.share_plans[season_id].append(plan)

    def generate_season_shares(self, subscription_id: str):
        """Generate all shares for a subscription based on schedule"""
        subscription = self.get_subscription(subscription_id)
        if not subscription:
            raise ValueError(f"Subscription {subscription_id} not found")

        if not subscription.delivery_schedule_id:
            raise ValueError("No delivery schedule for subscription")

        schedule = self.schedules.get(subscription.delivery_schedule_id)
        if not schedule:
            raise ValueError("Delivery schedule not found")

        # Generate share for each delivery date
        for delivery_date in schedule.delivery_dates:
            share = self.generate_share(subscription_id, delivery_date)
            subscription.shares.append(share.share_id)

    def generate_share(self, subscription_id: str, delivery_date: date) -> CSAShare:
        """Generate a single share for a subscription"""
        subscription = self.get_subscription(subscription_id)
        if not subscription:
            raise ValueError(f"Subscription {subscription_id} not found")

        # Find share plan for this date (if exists)
        season_id = f"season_{subscription.farm_id}_{subscription.start_date.year}_{subscription.season_type.value}"
        plans = self.share_plans.get(season_id, [])
        plan = next((p for p in plans if p.share_date == delivery_date), None)

        if plan:
            items = plan.items
            farmer_note = plan.notes
            recipes = plan.recipes
        else:
            # Default items (in production, would be more sophisticated)
            items = self._generate_default_items(subscription.share_size)
            farmer_note = f"Your {subscription.share_size.value} share for {delivery_date}"
            recipes = []

        # Create share
        share = create_csa_share(
            subscription.subscription_id,
            subscription.farm_id,
            delivery_date,
            items,
            farmer_note
        )

        share.recipe_suggestions = recipes

        # Store share
        self.shares[share.share_id] = share
        self.shares_by_subscription[subscription_id].append(share.share_id)

        return share

    def _generate_default_items(self, share_size: CSAShareSize) -> List[CSAShareItem]:
        """Generate default items based on share size"""
        # Base quantities
        multiplier = {
            CSAShareSize.MINI: Decimal("0.5"),
            CSAShareSize.SMALL: Decimal("0.75"),
            CSAShareSize.REGULAR: Decimal("1.0"),
            CSAShareSize.LARGE: Decimal("1.5"),
            CSAShareSize.FAMILY: Decimal("2.0")
        }.get(share_size, Decimal("1.0"))

        items = [
            CSAShareItem("Mixed Greens", Decimal("1.0") * multiplier, "lbs", Decimal("8.00")),
            CSAShareItem("Tomatoes", Decimal("2.0") * multiplier, "lbs", Decimal("10.00")),
            CSAShareItem("Carrots", Decimal("1.0") * multiplier, "bunches", Decimal("6.00")),
            CSAShareItem("Summer Squash", Decimal("1.5") * multiplier, "lbs", Decimal("7.00")),
            CSAShareItem("Fresh Herbs", Decimal("1.0") * multiplier, "bunches", Decimal("5.00"))
        ]

        return items

    def get_subscription_shares(
        self,
        subscription_id: str,
        delivered_only: bool = False
    ) -> List[CSAShare]:
        """Get all shares for a subscription"""
        share_ids = self.shares_by_subscription.get(subscription_id, [])
        shares = [self.shares[sid] for sid in share_ids if sid in self.shares]

        if delivered_only:
            shares = [s for s in shares if s.delivered]

        # Sort by delivery date
        shares.sort(key=lambda s: s.delivery_date)

        return shares

    def get_upcoming_shares(
        self,
        farm_id: Optional[str] = None,
        days_ahead: int = 7
    ) -> List[CSAShare]:
        """Get shares due for delivery in next N days"""
        today = date.today()
        cutoff = today + timedelta(days=days_ahead)

        all_shares = list(self.shares.values())
        upcoming = [
            s for s in all_shares
            if not s.delivered and today <= s.delivery_date <= cutoff
        ]

        if farm_id:
            upcoming = [s for s in upcoming if s.farm_id == farm_id]

        # Sort by delivery date
        upcoming.sort(key=lambda s: s.delivery_date)

        return upcoming

    def mark_share_delivered(self, share_id: str, rating: Optional[int] = None, feedback: str = ""):
        """Mark share as delivered"""
        share = self.shares.get(share_id)
        if not share:
            raise ValueError(f"Share {share_id} not found")

        share.mark_delivered()

        if rating:
            share.member_rating = rating
        if feedback:
            share.member_feedback = feedback

        # Update subscription
        subscription = self.get_subscription(share.subscription_id)
        if subscription:
            subscription.record_share_received(share_id)

    # ========================================================================
    # Delivery Locations
    # ========================================================================

    def add_pickup_location(
        self,
        schedule_id: str,
        location_id: str,
        name: str,
        address: str,
        city: str,
        state: str,
        zip_code: str,
        pickup_day: str,
        pickup_time_start: str,
        pickup_time_end: str,
        max_members: int = 50
    ) -> DeliveryLocation:
        """Add a pickup location to delivery schedule"""
        schedule = self.schedules.get(schedule_id)
        if not schedule:
            raise ValueError(f"Schedule {schedule_id} not found")

        location = DeliveryLocation(
            location_id=location_id,
            name=name,
            address=address,
            city=city,
            state=state,
            zip_code=zip_code,
            pickup_day=pickup_day,
            pickup_time_start=pickup_time_start,
            pickup_time_end=pickup_time_end,
            max_members=max_members
        )

        schedule.add_pickup_location(location)
        self.locations[location_id] = location

        return location

    def get_available_locations(self, schedule_id: str) -> List[DeliveryLocation]:
        """Get pickup locations with available space"""
        schedule = self.schedules.get(schedule_id)
        if not schedule:
            return []

        return [loc for loc in schedule.pickup_locations if not loc.is_full()]

    # ========================================================================
    # Analytics and Reporting
    # ========================================================================

    def get_csa_analytics(self, farm_id: str, season_id: str) -> CSAAnalytics:
        """Generate analytics for CSA program"""
        season = self.get_season(season_id)
        if not season:
            raise ValueError(f"Season {season_id} not found")

        subscriptions = self.get_season_subscriptions(season_id)

        analytics = CSAAnalytics(
            farm_id=farm_id,
            season_id=season_id,
            period_start=season.season_starts,
            period_end=season.season_ends
        )

        # Membership counts
        analytics.total_members = len(subscriptions)
        analytics.active_members = len([s for s in subscriptions if s.status == SubscriptionStatus.ACTIVE])
        analytics.paused_members = len([s for s in subscriptions if s.status == SubscriptionStatus.PAUSED])
        analytics.cancelled_members = len([s for s in subscriptions if s.status == SubscriptionStatus.CANCELLED])

        # Revenue
        analytics.total_revenue = sum(s.amount_paid for s in subscriptions)
        analytics.projected_revenue = sum(s.total_cost for s in subscriptions)

        if subscriptions:
            analytics.average_subscription_value = analytics.total_revenue / Decimal(str(len(subscriptions)))

        # Shares
        all_shares = []
        for sub in subscriptions:
            shares = self.get_subscription_shares(sub.subscription_id)
            all_shares.extend(shares)

        analytics.total_shares_delivered = len([s for s in all_shares if s.delivered])
        analytics.shares_scheduled = len(all_shares)

        if analytics.shares_scheduled > 0:
            analytics.delivery_completion_rate = (analytics.total_shares_delivered / analytics.shares_scheduled) * 100

        # Satisfaction
        ratings = [s.member_rating for s in all_shares if s.member_rating is not None]
        if ratings:
            analytics.average_rating = sum(ratings) / len(ratings)
            analytics.feedback_count = len([s for s in all_shares if s.member_feedback])

        return analytics

    def get_member_satisfaction_score(self, subscription_id: str) -> float:
        """Calculate satisfaction score for a member"""
        shares = self.get_subscription_shares(subscription_id, delivered_only=True)

        if not shares:
            return 0.0

        # Calculate based on ratings
        ratings = [s.member_rating for s in shares if s.member_rating is not None]

        if not ratings:
            return 0.0

        return sum(ratings) / len(ratings)

    def get_retention_metrics(self, farm_id: str) -> Dict:
        """Calculate member retention metrics"""
        # Get all subscriptions for farm
        all_subscriptions = []
        for season_ids in self.seasons_by_farm.get(farm_id, []):
            season_subs = self.get_season_subscriptions(season_ids)
            all_subscriptions.extend(season_subs)

        if not all_subscriptions:
            return {"retention_rate": 0.0, "churn_rate": 0.0, "renewal_count": 0}

        # Calculate retention
        completed = [s for s in all_subscriptions if s.status == SubscriptionStatus.COMPLETED]
        renewed = [s for s in completed if s.renewal_likely is True]

        retention_rate = (len(renewed) / len(completed) * 100) if completed else 0.0
        churn_rate = 100.0 - retention_rate

        return {
            "retention_rate": retention_rate,
            "churn_rate": churn_rate,
            "renewal_count": len(renewed),
            "completed_seasons": len(completed)
        }

    # ========================================================================
    # Member Communication
    # ========================================================================

    def send_share_notification(self, share_id: str) -> Dict:
        """Send notification about upcoming share"""
        share = self.shares.get(share_id)
        if not share:
            return {"success": False, "error": "Share not found"}

        subscription = self.get_subscription(share.subscription_id)
        if not subscription:
            return {"success": False, "error": "Subscription not found"}

        # In production, would integrate with email/SMS service
        notification = {
            "member_id": subscription.member_id,
            "subject": f"Your CSA share is ready for {share.delivery_date}",
            "message": f"""
            Hi there!

            Your {subscription.share_size.value} CSA share will be ready on {share.delivery_date}.

            This week's share includes:
            {self._format_share_items(share.items)}

            Farmer's Note: {share.farmer_note}

            {self._format_pickup_info(subscription)}

            Thank you for supporting local agriculture!
            """
        }

        return {"success": True, "notification": notification}

    def _format_share_items(self, items: List[CSAShareItem]) -> str:
        """Format share items for notification"""
        return "\n".join([f"- {item.item_name}: {item.quantity} {item.unit}" for item in items])

    def _format_pickup_info(self, subscription: CSASubscription) -> str:
        """Format pickup information"""
        if subscription.delivery_method == DeliveryMethod.FARM_PICKUP:
            return "Pickup at the farm during regular hours."
        elif subscription.delivery_method == DeliveryMethod.DROPOFF_LOCATION:
            if subscription.pickup_location_id:
                location = self.locations.get(subscription.pickup_location_id)
                if location:
                    return f"Pickup at {location.name}, {location.pickup_time_start}-{location.pickup_time_end}"
        elif subscription.delivery_method == DeliveryMethod.HOME_DELIVERY:
            return f"Delivery to: {subscription.home_delivery_address}"

        return "See your subscription details for pickup information."
