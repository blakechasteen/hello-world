#!/usr/bin/env python3
"""
Donation Coordinator - Intelligent matching between donors and recipients

Matches surplus food donations with organizations that need them based on:
- Geographic proximity
- Storage capabilities (refrigeration, freezing)
- Dietary requirements
- Capacity and need
- Urgency
- Historical success patterns
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from decimal import Decimal
import heapq

from sous.models.organization import Organization, FoodBank, Restaurant, GroceryStore
from sous.models.food_item import (
    FoodItem,
    DonationListing,
    SurplusAlert,
    StorageRequirement,
    DonationStatus
)


@dataclass
class MatchScore:
    """
    Score for a potential donor-recipient match.

    Higher scores indicate better matches.
    """
    recipient_org_id: str
    total_score: float
    subscores: Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""

    def __lt__(self, other):
        """Enable priority queue sorting (higher scores first)"""
        return self.total_score > other.total_score


@dataclass
class Match:
    """A confirmed match between donor and recipient"""
    match_id: str
    listing_id: str
    donor_org_id: str
    recipient_org_id: str
    matched_at: datetime
    score: float
    estimated_delivery_time: Optional[datetime] = None
    status: str = "pending"  # pending, confirmed, completed, cancelled


class DonationCoordinator:
    """
    Coordinates food donations between organizations.

    Uses intelligent matching algorithms to connect donors with
    the best recipient organizations.
    """

    def __init__(self):
        # Active listings and alerts
        self.active_listings: Dict[str, DonationListing] = {}
        self.active_alerts: Dict[str, SurplusAlert] = {}

        # Organization registry
        self.organizations: Dict[str, Organization] = {}

        # Match history for learning
        self.match_history: List[Match] = []
        self.successful_matches: Dict[Tuple[str, str], int] = {}  # (donor, recipient) -> count

        # Matching weights (can be tuned over time)
        self.weights = {
            "distance": 0.30,
            "storage_capability": 0.20,
            "capacity": 0.15,
            "urgency": 0.15,
            "historical_success": 0.10,
            "dietary_match": 0.10
        }

    # ========================================================================
    # Organization Management
    # ========================================================================

    def register_organization(self, org: Organization):
        """Register an organization in the network"""
        self.organizations[org.org_id] = org

    def get_organization(self, org_id: str) -> Optional[Organization]:
        """Get organization by ID"""
        return self.organizations.get(org_id)

    def get_donors(self) -> List[Organization]:
        """Get all organizations that can donate"""
        return [org for org in self.organizations.values() if org.can_donate and org.active]

    def get_recipients(self) -> List[Organization]:
        """Get all organizations that can receive"""
        return [org for org in self.organizations.values() if org.can_receive and org.active]

    # ========================================================================
    # Listing Management
    # ========================================================================

    def create_listing(self, listing: DonationListing) -> str:
        """
        Create a new donation listing.

        Returns listing ID.
        """
        self.active_listings[listing.listing_id] = listing

        # Automatically find matches for the listing
        matches = self.find_matches_for_listing(listing)

        # Notify top matches
        if matches:
            top_matches = matches[:5]  # Top 5 matches
            for match in top_matches:
                self._notify_organization(match.recipient_org_id, listing)

        return listing.listing_id

    def get_listing(self, listing_id: str) -> Optional[DonationListing]:
        """Get listing by ID"""
        return self.active_listings.get(listing_id)

    def get_available_listings(
        self,
        recipient_org_id: Optional[str] = None
    ) -> List[DonationListing]:
        """
        Get all available listings.

        Optionally filter for a specific recipient's needs.
        """
        listings = [
            listing for listing in self.active_listings.values()
            if listing.is_available()
        ]

        if recipient_org_id:
            # Filter listings compatible with recipient
            recipient = self.get_organization(recipient_org_id)
            if recipient:
                listings = [
                    listing for listing in listings
                    if self._is_compatible(listing, recipient)
                ]

        return listings

    def claim_listing(
        self,
        listing_id: str,
        recipient_org_id: str
    ) -> bool:
        """
        Claim a listing for a recipient.

        Returns True if successful.
        """
        listing = self.get_listing(listing_id)
        if not listing:
            return False

        if not listing.claim(recipient_org_id):
            return False

        # Record the match
        match = Match(
            match_id=f"match_{listing_id}_{recipient_org_id}",
            listing_id=listing_id,
            donor_org_id=listing.donor_org_id,
            recipient_org_id=recipient_org_id,
            matched_at=datetime.now(),
            score=0.0  # Will be calculated if needed
        )
        self.match_history.append(match)

        # Update successful match count
        key = (listing.donor_org_id, recipient_org_id)
        self.successful_matches[key] = self.successful_matches.get(key, 0) + 1

        return True

    # ========================================================================
    # Surplus Alerts
    # ========================================================================

    def create_alert(self, alert: SurplusAlert) -> str:
        """
        Create a surplus alert for immediate pickup needs.

        Returns alert ID.
        """
        self.active_alerts[alert.alert_id] = alert

        # Find nearby recipients who can respond quickly
        donor = self.get_organization(alert.donor_org_id)
        if not donor:
            return alert.alert_id

        nearby_recipients = self._find_nearby_recipients(
            donor,
            max_distance_miles=alert.items[0].storage_requirement == StorageRequirement.FROZEN and 3.0 or 10.0
        )

        # Notify all nearby recipients
        for recipient in nearby_recipients:
            alert.notified_org_ids.append(recipient.org_id)
            self._notify_organization(recipient.org_id, alert, urgent=True)

        return alert.alert_id

    def respond_to_alert(
        self,
        alert_id: str,
        recipient_org_id: str
    ) -> bool:
        """
        Respond to a surplus alert.

        Returns True if alert is still active.
        """
        alert = self.active_alerts.get(alert_id)
        if not alert or not alert.is_active():
            return False

        alert.respond(recipient_org_id)
        return True

    def claim_alert(
        self,
        alert_id: str,
        recipient_org_id: str
    ) -> bool:
        """
        Claim a surplus alert (first responder wins).

        Returns True if successful.
        """
        alert = self.active_alerts.get(alert_id)
        if not alert or not alert.is_active():
            return False

        alert.resolve(recipient_org_id)

        # Create a donation listing from the alert
        listing = DonationListing(
            listing_id=f"listing_from_alert_{alert_id}",
            donor_org_id=alert.donor_org_id,
            items=alert.items,
            available_from=datetime.now(),
            available_until=alert.expires_at or (datetime.now() + timedelta(hours=2)),
            status=DonationStatus.CLAIMED,
            claimed_by_org_id=recipient_org_id,
            claimed_at=datetime.now()
        )

        self.active_listings[listing.listing_id] = listing

        return True

    # ========================================================================
    # Intelligent Matching
    # ========================================================================

    def find_matches_for_listing(
        self,
        listing: DonationListing,
        max_matches: int = 10
    ) -> List[MatchScore]:
        """
        Find best recipient matches for a donation listing.

        Returns sorted list of matches (best first).
        """
        donor = self.get_organization(listing.donor_org_id)
        if not donor:
            return []

        recipients = self.get_recipients()
        if not recipients:
            return []

        # Score all recipients
        scores = []
        for recipient in recipients:
            score = self._calculate_match_score(listing, donor, recipient)
            if score.total_score > 0:
                scores.append(score)

        # Sort by score (highest first)
        scores.sort(reverse=True, key=lambda s: s.total_score)

        return scores[:max_matches]

    def _calculate_match_score(
        self,
        listing: DonationListing,
        donor: Organization,
        recipient: Organization
    ) -> MatchScore:
        """
        Calculate match score between donor and recipient.

        Considers multiple factors:
        - Geographic distance
        - Storage capabilities
        - Capacity to accept
        - Urgency
        - Historical success
        - Dietary requirements
        """
        subscores = {}

        # 1. Distance score (0-1, closer is better)
        distance = donor.get_distance_to(recipient.location)
        if distance is None:
            subscores["distance"] = 0.5  # Unknown distance
        elif distance > 50:
            subscores["distance"] = 0.0  # Too far
        else:
            subscores["distance"] = max(0, 1.0 - (distance / 50.0))

        # 2. Storage capability score (0-1)
        storage_score = 1.0
        for item in listing.items:
            if item.storage_requirement == StorageRequirement.REFRIGERATED:
                if not recipient.has_refrigeration:
                    storage_score = 0.0
                    break
            elif item.storage_requirement == StorageRequirement.FROZEN:
                if not recipient.has_freezer:
                    storage_score = 0.0
                    break
        subscores["storage_capability"] = storage_score

        # 3. Capacity score (0-1)
        if isinstance(recipient, FoodBank):
            capacity_percent = recipient.get_capacity_percent()
            # Lower capacity = higher score (more room to accept)
            subscores["capacity"] = max(0, 1.0 - (capacity_percent / 100.0))
        else:
            subscores["capacity"] = 0.7  # Assume moderate capacity for non-food banks

        # 4. Urgency score (0-1)
        days_until_expiration = min(
            item.get_days_until_expiration() or 999
            for item in listing.items
        )

        if days_until_expiration <= 1:
            subscores["urgency"] = 1.0  # Very urgent
        elif days_until_expiration <= 3:
            subscores["urgency"] = 0.7  # Somewhat urgent
        else:
            subscores["urgency"] = 0.4  # Not urgent

        # 5. Historical success score (0-1)
        key = (donor.org_id, recipient.org_id)
        past_successes = self.successful_matches.get(key, 0)
        # Normalize: 0 successes = 0.5, 10+ successes = 1.0
        subscores["historical_success"] = min(1.0, 0.5 + (past_successes / 20.0))

        # 6. Dietary match score (0-1)
        # For food banks serving specific dietary needs
        dietary_match = 1.0  # Default: all foods accepted
        subscores["dietary_match"] = dietary_match

        # Calculate weighted total score
        total_score = sum(
            subscores[key] * self.weights[key]
            for key in subscores
        )

        # Build reasoning
        reasoning_parts = []
        if subscores["distance"] > 0.7:
            reasoning_parts.append("Close proximity")
        if subscores["storage_capability"] == 1.0:
            reasoning_parts.append("Has required storage")
        if subscores["capacity"] > 0.7:
            reasoning_parts.append("Has capacity")
        if subscores["urgency"] > 0.7:
            reasoning_parts.append("Urgent need")
        if subscores["historical_success"] > 0.7:
            reasoning_parts.append("Good history")

        reasoning = ", ".join(reasoning_parts) if reasoning_parts else "Acceptable match"

        return MatchScore(
            recipient_org_id=recipient.org_id,
            total_score=total_score,
            subscores=subscores,
            reasoning=reasoning
        )

    def _is_compatible(
        self,
        listing: DonationListing,
        recipient: Organization
    ) -> bool:
        """Check if listing is compatible with recipient's capabilities"""
        # Check storage requirements
        for item in listing.items:
            if item.storage_requirement == StorageRequirement.REFRIGERATED:
                if not recipient.has_refrigeration:
                    return False
            elif item.storage_requirement == StorageRequirement.FROZEN:
                if not recipient.has_freezer:
                    return False

        # Check distance if delivery not available
        if not listing.delivery_available:
            donor = self.get_organization(listing.donor_org_id)
            if donor:
                distance = donor.get_distance_to(recipient.location)
                if distance and distance > 25:  # Must be within 25 miles for pickup
                    return False

        return True

    def _find_nearby_recipients(
        self,
        donor: Organization,
        max_distance_miles: float = 10.0
    ) -> List[Organization]:
        """Find recipients within distance of donor"""
        recipients = self.get_recipients()
        nearby = []

        for recipient in recipients:
            distance = donor.get_distance_to(recipient.location)
            if distance and distance <= max_distance_miles:
                nearby.append(recipient)

        return nearby

    def _notify_organization(
        self,
        org_id: str,
        item,  # DonationListing or SurplusAlert
        urgent: bool = False
    ):
        """
        Notify organization of new opportunity.

        In production, would send email/SMS/push notification.
        """
        org = self.get_organization(org_id)
        if not org:
            return

        # For now, just log
        notification_type = "URGENT" if urgent else "INFO"
        print(f"[{notification_type}] Notifying {org.name} about donation opportunity")

    # ========================================================================
    # Analytics
    # ========================================================================

    def get_matching_statistics(self) -> Dict:
        """Get statistics about matching performance"""
        total_matches = len(self.match_history)
        if total_matches == 0:
            return {
                "total_matches": 0,
                "success_rate": 0.0,
                "avg_time_to_claim_minutes": 0
            }

        # Count completed matches
        completed = sum(1 for m in self.match_history if m.status == "completed")

        # Calculate average time to claim
        claim_times = []
        for match in self.match_history:
            listing = self.get_listing(match.listing_id)
            if listing and listing.claimed_at:
                delta = (listing.claimed_at - listing.created_at).total_seconds() / 60
                claim_times.append(delta)

        avg_claim_time = sum(claim_times) / len(claim_times) if claim_times else 0

        return {
            "total_matches": total_matches,
            "completed_matches": completed,
            "success_rate": completed / total_matches if total_matches > 0 else 0.0,
            "avg_time_to_claim_minutes": avg_claim_time,
            "total_weight_matched_lbs": sum(
                self.get_listing(m.listing_id).get_total_weight_lbs()
                for m in self.match_history
                if self.get_listing(m.listing_id)
            )
        }

    def get_top_donor_recipient_pairs(self, limit: int = 10) -> List[Dict]:
        """Get most successful donor-recipient pairs"""
        pairs = [
            {
                "donor_org_id": donor_id,
                "recipient_org_id": recipient_id,
                "match_count": count
            }
            for (donor_id, recipient_id), count in self.successful_matches.items()
        ]

        pairs.sort(key=lambda p: p["match_count"], reverse=True)
        return pairs[:limit]
