"""Memory store interface and snapshot."""

from typing import Protocol, List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from ..domain import ElleRequest, ElleAction


@dataclass(frozen=True)
class InteractionRecord:
    """Single interaction: request + action + outcome."""
    
    timestamp: datetime
    request: ElleRequest
    action: ElleAction
    
    # Was the action accepted/completed?
    user_accepted: Optional[bool] = None
    task_completed: Optional[bool] = None
    completion_time: Optional[datetime] = None


@dataclass(frozen=True)
class UserPattern:
    """Learned patterns about user behavior."""
    
    # Temporal patterns
    most_active_times: List[str] = field(default_factory=list)  # ["morning", "afternoon"]
    typical_session_length: Optional[int] = None  # minutes
    
    # Task patterns
    preferred_task_size: str = "medium"  # "small", "medium", "large"
    task_completion_rate: float = 0.0  # 0-1
    
    # Interaction patterns
    response_to_suggestions: float = 0.5  # How often they act on suggestions (0-1)
    prefers_silence: bool = False  # True if they ignore most utterances


@dataclass(frozen=True)
class MemorySnapshot:
    """
    Read-only snapshot of user memory for decision-making.
    
    This is what gets passed to Elle Core policy.
    Frozen = immutable = cacheable.
    """
    
    user_id: str
    loaded_at: datetime
    
    # Recent context
    recent_interactions: List[InteractionRecord] = field(default_factory=list)
    recent_locations: List[str] = field(default_factory=list)
    
    # Long-term patterns
    patterns: Optional[UserPattern] = None
    
    # Active context
    active_tasks: List[str] = field(default_factory=list)  # Task IDs
    active_projects: List[str] = field(default_factory=list)  # Project IDs
    
    # Metadata
    total_interactions: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryStore(Protocol):
    """
    Interface for memory persistence.
    
    Different implementations can use different backends:
    - In-memory (for tests)
    - SQLite (local storage)
    - MirrorCore (when ready)
    """
    
    def load_memory(self, user_id: str) -> MemorySnapshot:
        """
        Load read-only snapshot for decision loop.
        
        This should be fast—no heavy computation.
        Pre-compute patterns offline if needed.
        """
        ...
    
    def record_interaction(
        self,
        request: ElleRequest,
        action: ElleAction
    ) -> None:
        """
        Record interaction for learning.
        
        Write happens after decision, doesn't block.
        """
        ...
    
    def update_task_outcome(
        self,
        task_id: str,
        completed: bool,
        completion_time: Optional[datetime] = None
    ) -> None:
        """Record task completion/cancellation."""
        ...
    
    def get_user_patterns(self, user_id: str) -> Optional[UserPattern]:
        """Get learned patterns for user."""
        ...
