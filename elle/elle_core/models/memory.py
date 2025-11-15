"""
Elle Core - Memory Models

Defines how Elle remembers past interactions and context.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum


class MemoryType(Enum):
    """Categories of memories Elle stores."""
    OBSERVATION = "observation"  # Something Elle observed
    INTERACTION = "interaction"  # User-Elle interaction
    TASK = "task"               # Task created or completed
    PROJECT = "project"         # Long-term project info
    PREFERENCE = "preference"   # Learned user preference
    CONTEXT = "context"         # Environmental context


@dataclass
class MemoryEntry:
    """
    A single memory - something Elle remembers.
    """
    id: str
    type: MemoryType
    content: str  # Natural language description
    timestamp: datetime = field(default_factory=datetime.now)
    location: Optional[str] = None  # Where this happened
    related_objects: List[str] = field(default_factory=list)  # Object IDs
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: float = 0.5  # 0.0-1.0, how important is this memory

    def __str__(self) -> str:
        time_str = self.timestamp.strftime("%Y-%m-%d %H:%M")
        return f"[{time_str}] {self.type.value}: {self.content}"


@dataclass
class MemorySnapshot:
    """
    A collection of relevant memories for a given context.

    This is what gets passed to Elle's policy to inform decisions.
    """
    recent: List[MemoryEntry] = field(default_factory=list)  # Recent memories
    relevant: List[MemoryEntry] = field(default_factory=list)  # Contextually relevant
    preferences: Dict[str, Any] = field(default_factory=dict)  # Known preferences

    def all_memories(self) -> List[MemoryEntry]:
        """Get all unique memories in this snapshot."""
        seen = set()
        result = []
        for memory in self.recent + self.relevant:
            if memory.id not in seen:
                result.append(memory)
                seen.add(memory.id)
        return result

    def get_memories_by_type(self, memory_type: MemoryType) -> List[MemoryEntry]:
        """Filter memories by type."""
        return [m for m in self.all_memories() if m.type == memory_type]

    def __str__(self) -> str:
        total = len(self.all_memories())
        prefs = len(self.preferences)
        return f"MemorySnapshot: {total} memories, {prefs} preferences"
