"""
Elle Core - Memory Service

Provides storage and retrieval of Elle's memories.
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging

from ..models import MemoryEntry, MemorySnapshot, MemoryType


logger = logging.getLogger(__name__)


class MemoryStore(ABC):
    """Abstract base class for memory storage backends."""

    @abstractmethod
    async def store(self, memory: MemoryEntry) -> None:
        """Store a memory."""
        pass

    @abstractmethod
    async def get_recent(self, limit: int = 10) -> List[MemoryEntry]:
        """Get recent memories."""
        pass

    @abstractmethod
    async def get_relevant(
        self,
        location: Optional[str] = None,
        related_objects: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """Get contextually relevant memories."""
        pass

    @abstractmethod
    async def get_preferences(self) -> Dict[str, Any]:
        """Get known user preferences."""
        pass

    async def get_snapshot(
        self,
        location: Optional[str] = None,
        related_objects: Optional[List[str]] = None,
    ) -> MemorySnapshot:
        """
        Get a memory snapshot for the current context.

        Combines recent and relevant memories.
        """
        recent = await self.get_recent(limit=5)
        relevant = await self.get_relevant(
            location=location,
            related_objects=related_objects,
            limit=5,
        )
        preferences = await self.get_preferences()

        return MemorySnapshot(
            recent=recent,
            relevant=relevant,
            preferences=preferences,
        )


class InMemoryStore(MemoryStore):
    """
    Simple in-memory storage for development/testing.

    This stores all memories in RAM and doesn't persist between sessions.
    For production, use a persistent backend (Neo4j, SQLite, etc.).
    """

    def __init__(self):
        self.memories: List[MemoryEntry] = []
        self.preferences: Dict[str, Any] = {}

    async def store(self, memory: MemoryEntry) -> None:
        """Store a memory in RAM."""
        self.memories.append(memory)
        logger.debug(f"Stored memory: {memory.id} ({memory.type.value})")

        # Extract preferences from memory
        if memory.type == MemoryType.PREFERENCE:
            # Simple key-value extraction from content
            # In production, this would be more sophisticated
            parts = memory.content.split(":", 1)
            if len(parts) == 2:
                key = parts[0].strip().lower().replace(" ", "_")
                value = parts[1].strip()
                self.preferences[key] = value

    async def get_recent(self, limit: int = 10) -> List[MemoryEntry]:
        """Get most recent memories."""
        # Sort by timestamp, newest first
        sorted_memories = sorted(
            self.memories,
            key=lambda m: m.timestamp,
            reverse=True,
        )
        return sorted_memories[:limit]

    async def get_relevant(
        self,
        location: Optional[str] = None,
        related_objects: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """
        Get contextually relevant memories.

        Filters by location and/or related objects.
        """
        relevant = []

        for memory in self.memories:
            # Filter by location
            if location and memory.location != location:
                continue

            # Filter by related objects
            if related_objects:
                # Check if any of the related objects match
                if not any(obj in memory.related_objects for obj in related_objects):
                    continue

            relevant.append(memory)

        # Sort by importance and recency
        relevant.sort(
            key=lambda m: (m.importance, m.timestamp),
            reverse=True,
        )

        return relevant[:limit]

    async def get_preferences(self) -> Dict[str, Any]:
        """Get known user preferences."""
        return self.preferences.copy()

    def clear(self) -> None:
        """Clear all memories (for testing)."""
        self.memories.clear()
        self.preferences.clear()


class MemoryService:
    """
    High-level memory service.

    Provides convenience methods for common memory operations.
    """

    def __init__(self, store: MemoryStore):
        self.store = store

    async def log_observation(
        self,
        content: str,
        location: Optional[str] = None,
        related_objects: Optional[List[str]] = None,
        importance: float = 0.5,
    ) -> MemoryEntry:
        """Log an observation."""
        memory = MemoryEntry(
            id=f"obs_{datetime.now().timestamp()}",
            type=MemoryType.OBSERVATION,
            content=content,
            location=location,
            related_objects=related_objects or [],
            importance=importance,
        )
        await self.store.store(memory)
        return memory

    async def log_interaction(
        self,
        content: str,
        location: Optional[str] = None,
        importance: float = 0.7,
    ) -> MemoryEntry:
        """Log a user-Elle interaction."""
        memory = MemoryEntry(
            id=f"int_{datetime.now().timestamp()}",
            type=MemoryType.INTERACTION,
            content=content,
            location=location,
            importance=importance,
        )
        await self.store.store(memory)
        return memory

    async def log_preference(
        self,
        preference: str,
        importance: float = 0.9,
    ) -> MemoryEntry:
        """Log a learned user preference."""
        memory = MemoryEntry(
            id=f"pref_{datetime.now().timestamp()}",
            type=MemoryType.PREFERENCE,
            content=preference,
            importance=importance,
        )
        await self.store.store(memory)
        return memory

    async def get_snapshot(
        self,
        location: Optional[str] = None,
        related_objects: Optional[List[str]] = None,
    ) -> MemorySnapshot:
        """Get a memory snapshot for decision-making."""
        return await self.store.get_snapshot(
            location=location,
            related_objects=related_objects,
        )
