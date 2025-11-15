"""
Baseline Behavior Modeler

Learns normal user behavior patterns for anomaly detection.

Features:
- Per-user and per-role baselines
- Rolling window (7+ days minimum)
- Incremental learning
- Tracks: request rate, endpoints, query complexity, time of day, location, etc.

Author: HoloLoom Security Team
Created: 2025-11-15
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from collections import defaultdict, Counter
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class UserBaseline:
    """
    Baseline behavior profile for a single user.

    Tracks typical patterns for anomaly detection.
    """
    user_id: str
    window_days: int = 7

    # Request patterns
    avg_requests_per_hour: float = 0.0
    std_requests_per_hour: float = 0.0
    peak_hour: Optional[int] = None

    # Temporal patterns
    typical_hours: Set[int] = field(default_factory=set)  # Hours user is typically active
    typical_days: Set[int] = field(default_factory=set)  # Days of week (0=Mon, 6=Sun)
    works_weekends: bool = False

    # Geographic patterns
    typical_location: Optional[str] = None  # Most common country code
    known_locations: Set[str] = field(default_factory=set)  # All observed locations

    # Access patterns
    typical_endpoints: Set[str] = field(default_factory=set)
    endpoint_frequencies: Dict[str, int] = field(default_factory=dict)

    # Query complexity
    avg_query_complexity: float = 0.5
    std_query_complexity: float = 0.1

    # Authentication patterns
    typical_auth_methods: Set[str] = field(default_factory=set)

    # Metadata
    first_seen: Optional[float] = None  # Unix timestamp
    last_seen: Optional[float] = None
    total_events: int = 0

    def update_from_events(self, events: List['SecurityEvent']):
        """
        Update baseline from a list of events (incremental learning).

        Args:
            events: List of SecurityEvent objects
        """
        if not events:
            return

        # Update temporal info
        self.first_seen = min(e.timestamp for e in events)
        self.last_seen = max(e.timestamp for e in events)
        self.total_events += len(events)

        # Temporal patterns
        hours = [e.hour_of_day for e in events if e.hour_of_day is not None]
        if hours:
            self.typical_hours.update(hours)
            hour_counts = Counter(hours)
            self.peak_hour = hour_counts.most_common(1)[0][0]

        days = [e.day_of_week for e in events if e.day_of_week is not None]
        if days:
            self.typical_days.update(days)
            self.works_weekends = any(d >= 5 for d in days)

        # Geographic patterns
        locations = [e.geographic_location for e in events if e.geographic_location]
        if locations:
            self.known_locations.update(locations)
            location_counts = Counter(locations)
            self.typical_location = location_counts.most_common(1)[0][0]

        # Access patterns
        endpoints = [e.endpoint for e in events if e.endpoint]
        if endpoints:
            self.typical_endpoints.update(endpoints)
            for endpoint, count in Counter(endpoints).items():
                self.endpoint_frequencies[endpoint] = \
                    self.endpoint_frequencies.get(endpoint, 0) + count

        # Query complexity
        complexities = [e.query_complexity for e in events if e.query_complexity is not None]
        if complexities:
            self.avg_query_complexity = float(np.mean(complexities))
            self.std_query_complexity = float(np.std(complexities))

        # Authentication patterns
        auth_methods = [e.authentication_method for e in events if e.authentication_method]
        if auth_methods:
            self.typical_auth_methods.update(auth_methods)

        # Request rate (events per hour)
        if self.first_seen and self.last_seen:
            time_span_hours = (self.last_seen - self.first_seen) / 3600.0
            if time_span_hours > 0:
                self.avg_requests_per_hour = len(events) / time_span_hours

                # Compute std by bucketing into hours
                hour_buckets = defaultdict(int)
                for event in events:
                    hour_bucket = int(event.timestamp // 3600)
                    hour_buckets[hour_bucket] += 1

                if hour_buckets:
                    self.std_requests_per_hour = float(np.std(list(hour_buckets.values())))

    def is_typical_behavior(self, event: 'SecurityEvent') -> bool:
        """
        Check if an event matches typical behavior.

        Args:
            event: SecurityEvent to check

        Returns:
            True if event is typical, False if anomalous
        """
        # Check temporal patterns
        if event.hour_of_day is not None and self.typical_hours:
            if event.hour_of_day not in self.typical_hours:
                return False

        # Check geographic location
        if event.geographic_location and self.known_locations:
            if event.geographic_location not in self.known_locations:
                return False

        # Check endpoint access
        if event.endpoint and self.typical_endpoints:
            if event.endpoint not in self.typical_endpoints:
                return False

        # Check query complexity (within 2 std devs)
        if event.query_complexity is not None:
            if abs(event.query_complexity - self.avg_query_complexity) > 2 * self.std_query_complexity:
                return False

        return True

    def get_deviation_score(self, event: 'SecurityEvent') -> float:
        """
        Compute deviation score for an event (0.0 = typical, 1.0 = very anomalous).

        Args:
            event: SecurityEvent to score

        Returns:
            Deviation score (0.0-1.0)
        """
        deviations = []

        # Temporal deviation
        if event.hour_of_day is not None and self.typical_hours:
            if event.hour_of_day not in self.typical_hours:
                deviations.append(1.0)
            else:
                deviations.append(0.0)

        # Geographic deviation
        if event.geographic_location and self.typical_location:
            if event.geographic_location != self.typical_location:
                deviations.append(0.8 if event.geographic_location in self.known_locations else 1.0)
            else:
                deviations.append(0.0)

        # Endpoint deviation
        if event.endpoint and self.typical_endpoints:
            if event.endpoint not in self.typical_endpoints:
                deviations.append(0.7)
            else:
                # Frequency-based score (rare endpoints get higher score)
                total_requests = sum(self.endpoint_frequencies.values())
                endpoint_freq = self.endpoint_frequencies.get(event.endpoint, 0)
                if total_requests > 0:
                    normalized_freq = endpoint_freq / total_requests
                    deviations.append(1.0 - normalized_freq)

        # Query complexity deviation (z-score)
        if event.query_complexity is not None and self.std_query_complexity > 0:
            z_score = abs(event.query_complexity - self.avg_query_complexity) / self.std_query_complexity
            complexity_deviation = min(z_score / 3.0, 1.0)  # Normalize to 0-1
            deviations.append(complexity_deviation)

        # Average deviations
        return float(np.mean(deviations)) if deviations else 0.0


class BaselineModeler:
    """
    Baseline behavior modeler for anomaly detection.

    Features:
    - Per-user baselines (separate profile for each user)
    - Per-role baselines (aggregate profile for roles)
    - Rolling window (keep only recent data)
    - Incremental learning (update baselines continuously)
    """

    def __init__(self, window_days: int = 7):
        """
        Initialize baseline modeler.

        Args:
            window_days: Days of data to keep in rolling window
        """
        self.window_days = window_days
        self.user_baselines: Dict[str, UserBaseline] = {}
        self.role_baselines: Dict[str, UserBaseline] = {}

        # Event history for rolling window
        self._event_history: List['SecurityEvent'] = []
        self._cutoff_timestamp: Optional[float] = None

        logger.info(f"BaselineModeler initialized (window_days={window_days})")

    async def fit(self, events: List['SecurityEvent'], per_user: bool = True):
        """
        Train baseline models on historical events.

        Args:
            events: List of security events (should be 7+ days of normal activity)
            per_user: Create separate baselines per user (recommended)
        """
        logger.info(f"Fitting baseline on {len(events)} events (per_user={per_user})")

        if not events:
            logger.warning("No events provided for baseline modeling")
            return

        # Store events in history
        self._event_history.extend(events)

        # Compute cutoff timestamp (keep only recent window)
        latest_timestamp = max(e.timestamp for e in events)
        self._cutoff_timestamp = latest_timestamp - (self.window_days * 86400)

        # Prune old events
        self._prune_old_events()

        if per_user:
            # Group events by user
            user_events = defaultdict(list)
            for event in events:
                user_events[event.user_id].append(event)

            # Create/update baseline for each user
            for user_id, user_event_list in user_events.items():
                if user_id not in self.user_baselines:
                    self.user_baselines[user_id] = UserBaseline(
                        user_id=user_id,
                        window_days=self.window_days
                    )
                self.user_baselines[user_id].update_from_events(user_event_list)

            logger.info(f"Created/updated baselines for {len(self.user_baselines)} users")

        else:
            # Create single global baseline
            if "global" not in self.user_baselines:
                self.user_baselines["global"] = UserBaseline(
                    user_id="global",
                    window_days=self.window_days
                )
            self.user_baselines["global"].update_from_events(events)
            logger.info("Created/updated global baseline")

    def update_incremental(self, event: 'SecurityEvent'):
        """
        Update baseline with a single new event (incremental learning).

        Args:
            event: New security event
        """
        # Add to history
        self._event_history.append(event)

        # Prune if needed
        if self._cutoff_timestamp:
            current_cutoff = event.timestamp - (self.window_days * 86400)
            if current_cutoff > self._cutoff_timestamp:
                self._cutoff_timestamp = current_cutoff
                self._prune_old_events()

        # Update user baseline
        if event.user_id not in self.user_baselines:
            self.user_baselines[event.user_id] = UserBaseline(
                user_id=event.user_id,
                window_days=self.window_days
            )
        self.user_baselines[event.user_id].update_from_events([event])

    def get_baseline(self, user_id: str) -> Optional[UserBaseline]:
        """
        Get baseline for a user.

        Args:
            user_id: User ID

        Returns:
            UserBaseline if exists, else None
        """
        return self.user_baselines.get(user_id)

    def _prune_old_events(self):
        """Remove events outside the rolling window."""
        if not self._cutoff_timestamp:
            return

        before_count = len(self._event_history)
        self._event_history = [
            e for e in self._event_history
            if e.timestamp >= self._cutoff_timestamp
        ]
        after_count = len(self._event_history)

        if before_count > after_count:
            logger.info(f"Pruned {before_count - after_count} old events from history")

    def get_statistics(self) -> Dict[str, any]:
        """
        Get baseline statistics.

        Returns:
            Dict with statistics
        """
        return {
            "window_days": self.window_days,
            "total_users": len(self.user_baselines),
            "total_events": len(self._event_history),
            "cutoff_timestamp": self._cutoff_timestamp,
            "avg_events_per_user": (
                len(self._event_history) / len(self.user_baselines)
                if self.user_baselines else 0
            )
        }
