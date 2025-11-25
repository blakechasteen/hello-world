#!/usr/bin/env python3
"""
Community Network - Organization connections and relationship management

Manages the social network of organizations in the food sharing ecosystem:
- Organization connections and partnerships
- Trust networks and verification
- Communication preferences
- Multi-org collaborations
- Network analytics and insights
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Tuple
from decimal import Decimal
from enum import Enum
from collections import defaultdict

from sous.models.organization import Organization


class ConnectionType(Enum):
    """Types of connections between organizations"""
    REGULAR_DONOR = "regular_donor"  # Org A regularly donates to Org B
    REGULAR_RECIPIENT = "regular_recipient"  # Org A regularly receives from Org B
    PARTNER = "partner"  # Mutual partnership
    VERIFIED = "verified"  # Verified trusted connection
    PREFERRED = "preferred"  # Preferred partner
    BLOCKED = "blocked"  # Blocked from connections


class CommunicationChannel(Enum):
    """Communication channels for notifications"""
    EMAIL = "email"
    SMS = "sms"
    PUSH_NOTIFICATION = "push"
    PHONE_CALL = "phone"
    IN_APP = "in_app"


class CollaborationType(Enum):
    """Types of multi-org collaborations"""
    FOOD_DRIVE = "food_drive"
    COMMUNITY_EVENT = "community_event"
    EMERGENCY_RESPONSE = "emergency_response"
    REGULAR_PICKUP = "regular_pickup"
    BULK_TRANSFER = "bulk_transfer"


@dataclass
class Connection:
    """Connection between two organizations"""
    from_org_id: str
    to_org_id: str
    connection_type: ConnectionType

    # Trust and verification
    verified: bool = False
    trust_score: float = 0.5  # 0.0-1.0

    # History
    established_at: datetime = field(default_factory=datetime.now)
    last_interaction: Optional[datetime] = None
    total_transactions: int = 0
    successful_transactions: int = 0

    # Metadata
    notes: str = ""
    tags: List[str] = field(default_factory=list)

    def get_success_rate(self) -> float:
        """Calculate success rate of transactions"""
        if self.total_transactions == 0:
            return 0.0
        return self.successful_transactions / self.total_transactions

    def update_trust_score(self, transaction_success: bool):
        """Update trust score based on transaction outcome"""
        # Exponential moving average with 0.1 learning rate
        if transaction_success:
            self.trust_score = self.trust_score * 0.9 + 0.1
            self.successful_transactions += 1
        else:
            self.trust_score = self.trust_score * 0.9

        self.total_transactions += 1
        self.last_interaction = datetime.now()

        # Clamp to [0, 1]
        self.trust_score = max(0.0, min(1.0, self.trust_score))


@dataclass
class CommunicationPreferences:
    """Organization's communication preferences"""
    org_id: str

    # Channels (ordered by preference)
    preferred_channels: List[CommunicationChannel] = field(default_factory=list)

    # Availability windows
    available_hours_start: str = "09:00"  # HH:MM
    available_hours_end: str = "17:00"
    available_days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon-Fri

    # Notification settings
    urgent_alerts: bool = True
    daily_digest: bool = True
    weekly_summary: bool = True

    # Contact info
    primary_contact_email: str = ""
    primary_contact_phone: str = ""

    # Response preferences
    auto_respond: bool = False
    response_time_hours: int = 24


@dataclass
class Collaboration:
    """Multi-organization collaboration"""
    collaboration_id: str
    name: str
    collaboration_type: CollaborationType

    # Participants
    participating_org_ids: List[str] = field(default_factory=list)
    lead_org_id: Optional[str] = None

    # Schedule
    start_date: datetime = field(default_factory=datetime.now)
    end_date: Optional[datetime] = None
    recurring: bool = False
    recurrence_pattern: Optional[str] = None  # e.g., "weekly", "monthly"

    # Status
    active: bool = True

    # Outcomes
    total_food_lbs: Decimal = field(default_factory=lambda: Decimal("0.0"))
    families_served: int = 0

    # Metadata
    description: str = ""
    goals: List[str] = field(default_factory=list)
    achievements: List[str] = field(default_factory=list)


class CommunityNetwork:
    """
    Manages the social network of organizations.

    Handles connections, partnerships, communication, and collaborations.
    """

    def __init__(self):
        # Organizations in the network
        self.organizations: Dict[str, Organization] = {}

        # Connections: (from_id, to_id) -> Connection
        self.connections: Dict[Tuple[str, str], Connection] = {}

        # Communication preferences: org_id -> CommunicationPreferences
        self.communication_prefs: Dict[str, CommunicationPreferences] = {}

        # Collaborations: collaboration_id -> Collaboration
        self.collaborations: Dict[str, Collaboration] = {}

        # Network metrics cache
        self._metrics_cache: Dict[str, any] = {}
        self._cache_timestamp: Optional[datetime] = None

    # ========================================================================
    # Organization Management
    # ========================================================================

    def add_organization(self, org: Organization):
        """Add organization to the network"""
        self.organizations[org.org_id] = org

        # Initialize default communication preferences
        if org.org_id not in self.communication_prefs:
            self.communication_prefs[org.org_id] = CommunicationPreferences(
                org_id=org.org_id,
                preferred_channels=[CommunicationChannel.EMAIL, CommunicationChannel.IN_APP],
                primary_contact_email=org.contact_email,
                primary_contact_phone=org.contact_phone
            )

    def get_organization(self, org_id: str) -> Optional[Organization]:
        """Get organization by ID"""
        return self.organizations.get(org_id)

    def remove_organization(self, org_id: str) -> bool:
        """Remove organization from network"""
        if org_id not in self.organizations:
            return False

        # Remove all connections
        keys_to_remove = [
            key for key in self.connections.keys()
            if key[0] == org_id or key[1] == org_id
        ]
        for key in keys_to_remove:
            del self.connections[key]

        # Remove from collaborations
        for collab in self.collaborations.values():
            if org_id in collab.participating_org_ids:
                collab.participating_org_ids.remove(org_id)

        # Remove org
        del self.organizations[org_id]
        self.communication_prefs.pop(org_id, None)

        return True

    # ========================================================================
    # Connection Management
    # ========================================================================

    def create_connection(
        self,
        from_org_id: str,
        to_org_id: str,
        connection_type: ConnectionType,
        verified: bool = False
    ) -> Connection:
        """Create connection between organizations"""
        key = (from_org_id, to_org_id)

        connection = Connection(
            from_org_id=from_org_id,
            to_org_id=to_org_id,
            connection_type=connection_type,
            verified=verified
        )

        self.connections[key] = connection
        return connection

    def get_connection(
        self,
        from_org_id: str,
        to_org_id: str
    ) -> Optional[Connection]:
        """Get connection between organizations"""
        return self.connections.get((from_org_id, to_org_id))

    def remove_connection(
        self,
        from_org_id: str,
        to_org_id: str
    ) -> bool:
        """Remove connection"""
        key = (from_org_id, to_org_id)
        if key in self.connections:
            del self.connections[key]
            return True
        return False

    def get_connections_for_org(
        self,
        org_id: str,
        direction: str = "both"  # "outgoing", "incoming", "both"
    ) -> List[Connection]:
        """Get all connections for an organization"""
        connections = []

        if direction in ["outgoing", "both"]:
            connections.extend([
                conn for (from_id, _), conn in self.connections.items()
                if from_id == org_id
            ])

        if direction in ["incoming", "both"]:
            connections.extend([
                conn for (_, to_id), conn in self.connections.items()
                if to_id == org_id
            ])

        return connections

    def get_trusted_partners(
        self,
        org_id: str,
        min_trust_score: float = 0.7
    ) -> List[str]:
        """Get trusted partner organization IDs"""
        connections = self.get_connections_for_org(org_id)

        trusted = [
            conn.to_org_id if conn.from_org_id == org_id else conn.from_org_id
            for conn in connections
            if conn.trust_score >= min_trust_score and conn.verified
        ]

        return trusted

    def verify_connection(
        self,
        from_org_id: str,
        to_org_id: str
    ) -> bool:
        """Mark connection as verified"""
        connection = self.get_connection(from_org_id, to_org_id)
        if connection:
            connection.verified = True
            return True
        return False

    # ========================================================================
    # Communication
    # ========================================================================

    def set_communication_preferences(
        self,
        org_id: str,
        preferences: CommunicationPreferences
    ):
        """Set organization's communication preferences"""
        self.communication_prefs[org_id] = preferences

    def get_communication_preferences(
        self,
        org_id: str
    ) -> Optional[CommunicationPreferences]:
        """Get organization's communication preferences"""
        return self.communication_prefs.get(org_id)

    def send_notification(
        self,
        org_id: str,
        message: str,
        urgent: bool = False
    ) -> bool:
        """
        Send notification to organization.

        In production, would route to actual communication channels.
        """
        prefs = self.get_communication_preferences(org_id)
        if not prefs:
            return False

        # Check if org accepts urgent alerts
        if urgent and not prefs.urgent_alerts:
            return False

        # Get preferred channel
        channel = prefs.preferred_channels[0] if prefs.preferred_channels else CommunicationChannel.EMAIL

        # Log notification (in production, send via actual channel)
        print(f"[{channel.value.upper()}] To {org_id}: {message}")

        return True

    def broadcast_to_network(
        self,
        message: str,
        org_ids: Optional[List[str]] = None,
        exclude_org_ids: Optional[List[str]] = None
    ):
        """Broadcast message to multiple organizations"""
        target_orgs = org_ids if org_ids else list(self.organizations.keys())

        if exclude_org_ids:
            target_orgs = [org_id for org_id in target_orgs if org_id not in exclude_org_ids]

        for org_id in target_orgs:
            self.send_notification(org_id, message)

    # ========================================================================
    # Collaborations
    # ========================================================================

    def create_collaboration(
        self,
        name: str,
        collaboration_type: CollaborationType,
        participating_org_ids: List[str],
        lead_org_id: Optional[str] = None,
        description: str = ""
    ) -> Collaboration:
        """Create multi-org collaboration"""
        collab_id = f"collab_{name.lower().replace(' ', '_')}_{datetime.now().timestamp()}"

        collaboration = Collaboration(
            collaboration_id=collab_id,
            name=name,
            collaboration_type=collaboration_type,
            participating_org_ids=participating_org_ids,
            lead_org_id=lead_org_id,
            description=description
        )

        self.collaborations[collab_id] = collaboration

        # Notify participants
        for org_id in participating_org_ids:
            self.send_notification(
                org_id,
                f"You've been added to collaboration: {name}"
            )

        return collaboration

    def get_collaboration(self, collaboration_id: str) -> Optional[Collaboration]:
        """Get collaboration by ID"""
        return self.collaborations.get(collaboration_id)

    def get_collaborations_for_org(self, org_id: str) -> List[Collaboration]:
        """Get all collaborations for an organization"""
        return [
            collab for collab in self.collaborations.values()
            if org_id in collab.participating_org_ids
        ]

    def end_collaboration(self, collaboration_id: str) -> bool:
        """End a collaboration"""
        collaboration = self.get_collaboration(collaboration_id)
        if not collaboration:
            return False

        collaboration.active = False
        collaboration.end_date = datetime.now()

        return True

    # ========================================================================
    # Network Analytics
    # ========================================================================

    def get_network_stats(self) -> Dict:
        """Get comprehensive network statistics"""
        # Use cache if fresh (< 5 minutes old)
        if (self._cache_timestamp and
            (datetime.now() - self._cache_timestamp).total_seconds() < 300):
            return self._metrics_cache

        total_orgs = len(self.organizations)
        total_connections = len(self.connections)

        # Count by org type
        by_type = defaultdict(int)
        for org in self.organizations.values():
            by_type[org.org_type.value] += 1

        # Connection metrics
        verified_connections = sum(1 for conn in self.connections.values() if conn.verified)
        avg_trust_score = (
            sum(conn.trust_score for conn in self.connections.values()) / total_connections
            if total_connections > 0 else 0.0
        )

        # Active collaborations
        active_collabs = sum(1 for collab in self.collaborations.values() if collab.active)

        # Network density (connections / possible connections)
        possible_connections = total_orgs * (total_orgs - 1)
        network_density = (
            total_connections / possible_connections
            if possible_connections > 0 else 0.0
        )

        stats = {
            "total_organizations": total_orgs,
            "organizations_by_type": dict(by_type),
            "total_connections": total_connections,
            "verified_connections": verified_connections,
            "average_trust_score": avg_trust_score,
            "active_collaborations": active_collabs,
            "network_density": network_density,
            "timestamp": datetime.now().isoformat()
        }

        # Cache results
        self._metrics_cache = stats
        self._cache_timestamp = datetime.now()

        return stats

    def get_most_connected_orgs(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get most connected organizations"""
        connection_counts = defaultdict(int)

        for (from_id, to_id) in self.connections.keys():
            connection_counts[from_id] += 1
            connection_counts[to_id] += 1

        # Sort by count
        sorted_orgs = sorted(
            connection_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_orgs[:limit]

    def get_strongest_partnerships(self, limit: int = 10) -> List[Dict]:
        """Get strongest partnerships (highest trust + most transactions)"""
        partnerships = []

        for (from_id, to_id), conn in self.connections.items():
            strength = (
                conn.trust_score * 0.6 +
                (conn.get_success_rate() * 0.3) +
                (min(conn.total_transactions, 100) / 100.0 * 0.1)
            )

            partnerships.append({
                "from_org_id": from_id,
                "to_org_id": to_id,
                "strength": strength,
                "trust_score": conn.trust_score,
                "success_rate": conn.get_success_rate(),
                "total_transactions": conn.total_transactions,
                "verified": conn.verified
            })

        partnerships.sort(key=lambda x: x["strength"], reverse=True)
        return partnerships[:limit]

    def find_connection_path(
        self,
        from_org_id: str,
        to_org_id: str,
        max_hops: int = 3
    ) -> Optional[List[str]]:
        """
        Find shortest connection path between organizations.

        Uses BFS to find path through trusted connections.
        """
        if from_org_id == to_org_id:
            return [from_org_id]

        # BFS
        queue = [(from_org_id, [from_org_id])]
        visited = {from_org_id}

        while queue:
            current, path = queue.pop(0)

            if len(path) > max_hops:
                continue

            # Get connected orgs
            connections = self.get_connections_for_org(current, direction="outgoing")

            for conn in connections:
                next_org = conn.to_org_id

                if next_org == to_org_id:
                    return path + [next_org]

                if next_org not in visited:
                    visited.add(next_org)
                    queue.append((next_org, path + [next_org]))

        return None  # No path found

    def suggest_connections(
        self,
        org_id: str,
        limit: int = 5
    ) -> List[Dict]:
        """
        Suggest potential connections based on:
        - Geographic proximity
        - Complementary capabilities (donors ↔ recipients)
        - Shared partners
        """
        org = self.get_organization(org_id)
        if not org:
            return []

        existing_connections = {
            conn.to_org_id for conn in self.get_connections_for_org(org_id, "outgoing")
        }

        suggestions = []

        for other_id, other_org in self.organizations.items():
            if other_id == org_id or other_id in existing_connections:
                continue

            score = 0.0
            reasons = []

            # 1. Geographic proximity
            distance = org.get_distance_to(other_org.location)
            if distance and distance < 10:
                score += 0.4
                reasons.append(f"Close proximity ({distance:.1f} miles)")

            # 2. Complementary capabilities
            if org.can_donate and other_org.can_receive:
                score += 0.3
                reasons.append("Complementary donor/recipient")

            # 3. Shared partners
            org_partners = set(
                conn.to_org_id for conn in self.get_connections_for_org(org_id)
            )
            other_partners = set(
                conn.to_org_id for conn in self.get_connections_for_org(other_id)
            )
            shared = len(org_partners & other_partners)
            if shared > 0:
                score += 0.2 * min(shared / 3.0, 1.0)
                reasons.append(f"{shared} shared partners")

            # 4. Similar organization types
            if org.org_type == other_org.org_type:
                score += 0.1
                reasons.append("Similar organization type")

            if score > 0:
                suggestions.append({
                    "org_id": other_id,
                    "org_name": other_org.name,
                    "score": score,
                    "reasons": reasons
                })

        suggestions.sort(key=lambda x: x["score"], reverse=True)
        return suggestions[:limit]

    # ========================================================================
    # Reporting
    # ========================================================================

    def generate_network_report(self) -> Dict:
        """Generate comprehensive network report"""
        stats = self.get_network_stats()
        most_connected = self.get_most_connected_orgs(5)
        strongest_partnerships = self.get_strongest_partnerships(5)

        return {
            "report_generated_at": datetime.now().isoformat(),
            "network_overview": stats,
            "most_connected_organizations": [
                {
                    "org_id": org_id,
                    "org_name": self.organizations[org_id].name,
                    "connection_count": count
                }
                for org_id, count in most_connected
            ],
            "strongest_partnerships": strongest_partnerships,
            "active_collaborations": len([
                c for c in self.collaborations.values() if c.active
            ]),
            "total_collaborations": len(self.collaborations)
        }
