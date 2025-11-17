"""
Shared Memory Protocol: Elle ↔ Proto Bridge via HoloLoom

This module provides the shared memory protocol that allows Elle (AR observer)
and Proto (Matrix bot) to communicate bidirectionally through HoloLoom's
knowledge graph.

Architecture:
    Elle observes physical space → stores in HoloLoom
    → HoloLoom triggers notification → Proto retrieves
    → Proto posts formatted update to Matrix

Design Principles:
- Event-driven: Elle pushes observations, Proto reacts
- Decoupled: Neither system depends on the other
- Async: All operations non-blocking
- Type-safe: Protocol-based with clear contracts

Created: 2025-11-17 (Week 4: Elle AR Bridge Integration)
"""

from typing import Optional, Dict, List, Any, Protocol, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import asyncio
import json

from HoloLoom import HoloLoom
from HoloLoom.config import Config


# =============================================================================
# Data Models for Elle Observations
# =============================================================================


@dataclass(frozen=True)
class ElleObservation:
    """
    Immutable record of Elle's observation in physical space.

    This is what gets stored in HoloLoom and retrieved by Proto.
    """

    # Identity
    observation_id: str
    timestamp: datetime

    # Location context
    location: str  # "workshop", "garden", "shed"

    # What Elle saw
    scene_summary: str  # "Workbench cleared, tools organized"
    objects_observed: List[str]  # ["shop_vac", "hand_tools", "workbench"]

    # Elle's assessment
    action_taken: Optional[str] = None  # "Suggested clearing workbench"
    user_response: Optional[str] = None  # "Completed suggestion"

    # Deferred items (for later)
    deferred_tasks: List[Dict[str, Any]] = field(default_factory=list)

    # Optional rich data
    photo_ids: List[str] = field(default_factory=list)  # Links to HoloLoom photos
    tags: List[str] = field(default_factory=list)  # ["productive", "decluttering"]

    # Proto notification
    notify_proto: bool = True  # Should Proto be notified?

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dict for HoloLoom storage."""
        return {
            'type': 'elle_observation',
            'observation_id': self.observation_id,
            'timestamp': self.timestamp.isoformat(),
            'location': self.location,
            'scene_summary': self.scene_summary,
            'objects_observed': self.objects_observed,
            'action_taken': self.action_taken,
            'user_response': self.user_response,
            'deferred_tasks': self.deferred_tasks,
            'photo_ids': self.photo_ids,
            'tags': self.tags,
            'notify_proto': self.notify_proto,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ElleObservation':
        """Reconstruct from HoloLoom data."""
        return cls(
            observation_id=data['observation_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            location=data['location'],
            scene_summary=data['scene_summary'],
            objects_observed=data['objects_observed'],
            action_taken=data.get('action_taken'),
            user_response=data.get('user_response'),
            deferred_tasks=data.get('deferred_tasks', []),
            photo_ids=data.get('photo_ids', []),
            tags=data.get('tags', []),
            notify_proto=data.get('notify_proto', True),
            metadata=data.get('metadata', {}),
        )


@dataclass(frozen=True)
class ProtoQuery:
    """
    Request from Proto to query Elle's observations.

    Proto can ask for:
    - Recent observations at a location
    - Observations with specific tags
    - Deferred tasks that need attention
    """

    query_id: str
    timestamp: datetime

    # Query type
    query_type: str  # "location", "tags", "deferred_tasks", "recent"

    # Parameters
    location: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    time_range_hours: Optional[int] = None  # Last N hours
    limit: int = 10

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Notification Protocol
# =============================================================================


class NotificationCallback(Protocol):
    """Protocol for Proto's notification handler."""

    async def __call__(self, observation: ElleObservation) -> None:
        """Handle new Elle observation."""
        ...


# =============================================================================
# Shared Memory Bridge
# =============================================================================


class ElleProtoMemoryBridge:
    """
    Shared memory bridge between Elle and Proto via HoloLoom.

    Responsibilities:
    - Store Elle observations in HoloLoom knowledge graph
    - Notify Proto when new observations arrive
    - Provide query interface for Proto
    - Maintain observation history

    Usage (Elle side):
        bridge = ElleProtoMemoryBridge()
        await bridge.store_observation(observation)

    Usage (Proto side):
        bridge = ElleProtoMemoryBridge()
        bridge.register_notification_handler(handle_elle_observation)
        observations = await bridge.query_observations(
            query_type="location",
            location="workshop"
        )
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        notification_handler: Optional[NotificationCallback] = None
    ):
        """
        Initialize shared memory bridge.

        Args:
            config: HoloLoom configuration (defaults to Config.fast())
            notification_handler: Callback for new observations (Proto side)
        """
        self.config = config or Config.fast()
        self._notification_handler = notification_handler
        self._loom: Optional[HoloLoom] = None

    async def __aenter__(self):
        """Async context manager entry."""
        # Initialize HoloLoom
        self._loom = HoloLoom(config=self.config)
        await self._loom.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._loom:
            await self._loom.__aexit__(exc_type, exc_val, exc_tb)

    # =========================================================================
    # Elle Side: Store Observations
    # =========================================================================

    async def store_observation(
        self,
        observation: ElleObservation
    ) -> str:
        """
        Store Elle observation in HoloLoom.

        Args:
            observation: ElleObservation to store

        Returns:
            Memory ID in HoloLoom

        Example:
            >>> observation = ElleObservation(
            ...     observation_id="obs_123",
            ...     timestamp=datetime.now(),
            ...     location="workshop",
            ...     scene_summary="Workbench cleared",
            ...     objects_observed=["shop_vac", "tools"],
            ... )
            >>> memory_id = await bridge.store_observation(observation)
        """
        if not self._loom:
            raise RuntimeError("Bridge not initialized. Use async context manager.")

        # Convert observation to dict
        obs_data = observation.to_dict()

        # Store in HoloLoom
        memory = await self._loom.experience(
            obs_data,
            context={
                'source': 'elle',
                'type': 'observation',
                'location': observation.location,
                'timestamp': observation.timestamp.isoformat(),
            }
        )

        # Trigger notification if requested
        if observation.notify_proto and self._notification_handler:
            await self._notification_handler(observation)

        return memory.id

    async def link_photo_to_observation(
        self,
        observation_id: str,
        photo_path: str,
        caption: Optional[str] = None
    ) -> str:
        """
        Link a photo to an Elle observation.

        Args:
            observation_id: ID of observation to link to
            photo_path: Path to photo file
            caption: Optional caption

        Returns:
            Photo token ID

        Example:
            >>> # Store observation
            >>> obs_id = await bridge.store_observation(observation)
            >>>
            >>> # Link photo
            >>> photo_id = await bridge.link_photo_to_observation(
            ...     obs_id,
            ...     "workshop_after.png",
            ...     caption="Workbench cleared and organized"
            ... )
        """
        if not self._loom:
            raise RuntimeError("Bridge not initialized. Use async context manager.")

        # Store photo in HoloLoom
        photo = await self._loom.remember_photo(
            photo_path,
            caption=caption,
            tags=['elle_observation'],
        )

        # Link to observation memory (if we can find it)
        # Note: Would need to search for observation by ID
        # For now, just return photo ID

        return photo.token_id

    # =========================================================================
    # Proto Side: Query Observations
    # =========================================================================

    async def query_observations(
        self,
        query_type: str = "recent",
        location: Optional[str] = None,
        tags: Optional[List[str]] = None,
        time_range_hours: Optional[int] = None,
        limit: int = 10
    ) -> List[ElleObservation]:
        """
        Query Elle observations from HoloLoom.

        Args:
            query_type: Type of query ("recent", "location", "tags", "deferred")
            location: Filter by location (e.g., "workshop")
            tags: Filter by tags (e.g., ["productive", "urgent"])
            time_range_hours: Only include observations from last N hours
            limit: Maximum observations to return

        Returns:
            List of ElleObservation objects

        Example:
            >>> # Get recent workshop observations
            >>> observations = await bridge.query_observations(
            ...     query_type="location",
            ...     location="workshop",
            ...     limit=5
            ... )
            >>>
            >>> # Get observations with specific tags
            >>> urgent = await bridge.query_observations(
            ...     query_type="tags",
            ...     tags=["urgent", "safety"],
            ... )
        """
        if not self._loom:
            raise RuntimeError("Bridge not initialized. Use async context manager.")

        # Build query based on type
        if query_type == "location" and location:
            query_text = f"Elle observations in {location}"
        elif query_type == "tags" and tags:
            tags_str = ", ".join(tags)
            query_text = f"Elle observations tagged: {tags_str}"
        elif query_type == "deferred":
            query_text = "Elle observations with deferred tasks"
        else:  # "recent"
            query_text = "Recent Elle observations"

        # Query HoloLoom
        memories = await self._loom.recall(query_text, limit=limit)

        # Filter and convert to ElleObservation objects
        observations = []
        for memory in memories:
            # Check if this is an Elle observation
            if memory.context.get('source') == 'elle' and memory.context.get('type') == 'observation':
                try:
                    # Parse the text back to dict (HoloLoom may have serialized it)
                    # For now, assume memory metadata contains the observation
                    obs_dict = memory.metadata

                    # Apply filters
                    if location and obs_dict.get('location') != location:
                        continue

                    if tags:
                        obs_tags = obs_dict.get('tags', [])
                        if not any(tag in obs_tags for tag in tags):
                            continue

                    if time_range_hours:
                        obs_time = datetime.fromisoformat(obs_dict['timestamp'])
                        cutoff = datetime.now() - timedelta(hours=time_range_hours)
                        if obs_time < cutoff:
                            continue

                    # Reconstruct observation
                    observation = ElleObservation.from_dict(obs_dict)
                    observations.append(observation)

                except Exception:
                    # Skip malformed observations
                    continue

        return observations

    async def get_deferred_tasks(
        self,
        location: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all deferred tasks from Elle observations.

        Args:
            location: Filter by location (optional)

        Returns:
            List of deferred task dicts

        Example:
            >>> tasks = await bridge.get_deferred_tasks("workshop")
            >>> for task in tasks:
            ...     print(f"{task['description']} (reason: {task['reason']})")
        """
        observations = await self.query_observations(
            query_type="deferred",
            location=location,
            limit=50  # Get more to find all deferred tasks
        )

        # Extract all deferred tasks
        tasks = []
        for obs in observations:
            for task in obs.deferred_tasks:
                # Add observation context
                task_with_context = {
                    **task,
                    'observation_id': obs.observation_id,
                    'location': obs.location,
                    'observed_at': obs.timestamp,
                }
                tasks.append(task_with_context)

        return tasks

    async def get_location_summary(
        self,
        location: str,
        time_range_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get summary of recent activity at a location.

        Args:
            location: Location to summarize (e.g., "workshop")
            time_range_hours: Look back this many hours

        Returns:
            Summary dict with stats and recent observations

        Example:
            >>> summary = await bridge.get_location_summary("workshop", 24)
            >>> print(f"Observations: {summary['observation_count']}")
            >>> print(f"Deferred tasks: {summary['deferred_tasks_count']}")
        """
        observations = await self.query_observations(
            query_type="location",
            location=location,
            time_range_hours=time_range_hours,
            limit=100
        )

        # Calculate statistics
        total_observations = len(observations)
        total_deferred = sum(len(obs.deferred_tasks) for obs in observations)
        total_actions = sum(1 for obs in observations if obs.action_taken)
        total_photos = sum(len(obs.photo_ids) for obs in observations)

        # Collect all tags
        all_tags = []
        for obs in observations:
            all_tags.extend(obs.tags)

        # Tag frequency
        from collections import Counter
        tag_counts = Counter(all_tags)

        return {
            'location': location,
            'time_range_hours': time_range_hours,
            'observation_count': total_observations,
            'deferred_tasks_count': total_deferred,
            'actions_taken': total_actions,
            'photos_count': total_photos,
            'top_tags': tag_counts.most_common(5),
            'recent_observations': [
                {
                    'timestamp': obs.timestamp,
                    'summary': obs.scene_summary,
                    'action': obs.action_taken,
                    'deferred': len(obs.deferred_tasks),
                }
                for obs in observations[:5]  # Most recent 5
            ],
        }

    # =========================================================================
    # Notification System
    # =========================================================================

    def register_notification_handler(
        self,
        handler: NotificationCallback
    ) -> None:
        """
        Register callback for Proto to receive Elle notifications.

        Args:
            handler: Async function to call when new observation arrives

        Example:
            >>> async def handle_observation(obs: ElleObservation):
            ...     print(f"Elle observed: {obs.scene_summary}")
            >>>
            >>> bridge.register_notification_handler(handle_observation)
        """
        self._notification_handler = handler

    # =========================================================================
    # Health Check
    # =========================================================================

    async def health_check(self) -> Dict[str, Any]:
        """
        Check bridge health.

        Returns:
            Health status dict
        """
        if not self._loom:
            return {
                'status': 'not_initialized',
                'loom': None,
            }

        try:
            metrics = self._loom.get_metrics()
            return {
                'status': 'healthy',
                'loom': {
                    'memories': metrics['n_memories'],
                    'active': metrics['n_active'],
                },
                'notification_handler_registered': self._notification_handler is not None,
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
            }


# =============================================================================
# Utility Functions
# =============================================================================


async def create_bridge(
    config: Optional[Config] = None,
    notification_handler: Optional[NotificationCallback] = None
) -> ElleProtoMemoryBridge:
    """
    Factory function to create and initialize bridge.

    Args:
        config: HoloLoom configuration
        notification_handler: Callback for notifications

    Returns:
        Initialized ElleProtoMemoryBridge (use with async context manager)

    Example:
        >>> async with create_bridge() as bridge:
        ...     await bridge.store_observation(observation)
    """
    bridge = ElleProtoMemoryBridge(config=config, notification_handler=notification_handler)
    return bridge


# =============================================================================
# Example Usage
# =============================================================================


if __name__ == "__main__":
    import asyncio
    from datetime import timedelta

    async def demo():
        """Demonstrate Elle → Proto bridge."""

        # Create bridge
        async with create_bridge() as bridge:
            print("🔗 Bridge initialized")

            # Elle stores observation
            observation = ElleObservation(
                observation_id="obs_workshop_001",
                timestamp=datetime.now(),
                location="workshop",
                scene_summary="Workbench cleared, tools organized",
                objects_observed=["shop_vac", "hand_tools", "workbench"],
                action_taken="Suggested clearing sawdust",
                user_response="Completed task",
                deferred_tasks=[
                    {
                        'description': 'Birdhouse project',
                        'reason': 'Need cedar boards',
                        'elle_note': 'Check for brad nails in toolbox',
                    }
                ],
                tags=['productive', 'decluttering'],
            )

            memory_id = await bridge.store_observation(observation)
            print(f"✅ Stored observation: {memory_id}")

            # Proto queries observations
            print("\n📊 Querying observations...")

            observations = await bridge.query_observations(
                query_type="location",
                location="workshop",
                limit=5
            )

            print(f"Found {len(observations)} observations:")
            for obs in observations:
                print(f"  • {obs.timestamp}: {obs.scene_summary}")

            # Get location summary
            summary = await bridge.get_location_summary("workshop", 24)
            print(f"\n📈 Workshop summary (last 24h):")
            print(f"  Observations: {summary['observation_count']}")
            print(f"  Deferred tasks: {summary['deferred_tasks_count']}")
            print(f"  Actions: {summary['actions_taken']}")

            # Get deferred tasks
            tasks = await bridge.get_deferred_tasks("workshop")
            print(f"\n⏸️ Deferred tasks ({len(tasks)}):")
            for task in tasks:
                print(f"  • {task['description']}")
                print(f"    Reason: {task['reason']}")
                print(f"    Note: {task.get('elle_note', 'N/A')}")

    # Run demo
    asyncio.run(demo())
